import copy
import functools
import os
import time
from types import SimpleNamespace

import wandb
import eval_humanml


from generate_utils import vis_during_train
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from tqdm import tqdm

from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
import eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader


def is_rank_zero():
    return ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0)


class TrainLoop_Flow:
    def __init__(self, cfg, train_platform, model, dynamic, data_loader, fixed_noise):
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.train_platform = train_platform
        self.model = model
        self.dynamic = dynamic
        self.cond_mode = model.cond_mode
        self.dataloader_train = data_loader
        self.batch_size = self.microbatch = cfg.batch_size
        self.fixed_noise = fixed_noise

        self.lr = cfg.training.lr

        self.log_interval = cfg.training.log_interval
        self.save_interval = cfg.training.save_interval
        self.resume_checkpoint = cfg.training.resume_checkpoint
        self.weight_decay = cfg.training.weight_decay
        self.lr_anneal_steps = cfg.training.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = cfg.training.num_steps
        self.num_epochs = self.num_steps // len(self.dataloader_train) + 1

        self._load_and_sync_parameters()

        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = cfg.training.save_dir
        self.overwrite = cfg.training.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())
        print("device: ", self.device)

        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if cfg.dataset in ["kit", "humanml"] and cfg.training.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(
                name=cfg.dataset,
                batch_size=cfg.training.eval_batch_size,
                num_frames=None,
                split=cfg.training.eval_split,
                hml_mode="eval",
            )

            self.eval_gt_data = get_dataset_loader(
                name=cfg.dataset,
                batch_size=cfg.training.eval_batch_size,
                num_frames=None,
                split=cfg.training.eval_split,
                hml_mode="gt",
            )
            self.eval_wrapper = EvaluatorMDMWrapper(cfg.dataset, dist_util.dev())
            self.eval_data = {
                "test": lambda: eval_humanml.get_mdm_loader(
                    model,
                    dynamic,
                    cfg.training.eval_batch_size,
                    gen_loader,
                    mm_num_samples,
                    mm_num_repeats,
                    gen_loader.dataset.opt.max_motion_length,
                    cfg.training.eval_num_samples,
                    scale=1.0,
                )
            }

        if torch.cuda.is_available():
            self.use_ddp = True
            print(dist_util.dev())
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            print("DDP is used")
        else:
            raise RuntimeError("CUDA is not available!")
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            for motion_feat, model_kwargs in tqdm(self.dataloader_train):
                if not (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
                ):
                    break

                motion_feat = motion_feat.to(self.device)
                model_kwargs["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in model_kwargs["y"].items()
                }

                self.run_step(motion_feat, model_kwargs)

                if self.step % self.log_interval == 0 and is_rank_zero():
                    for k, v in logger.get_current().name2val.items():
                        if k == "loss":
                            print(
                                "step[{}]: loss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )

                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k, value=v, iteration=self.step, group_name="Loss"
                            )

                if self.step % self.save_interval == 0 and is_rank_zero():
                    self.save()
                    self.model.eval()
                    curveness = self.dynamic.cal_curveness(
                        model=self.model,
                        z_orig=self.fixed_noise,
                        N=1000,
                        model_kwargs=model_kwargs,
                    )
                    self.train_platform.report_scalar(
                        name="curveness",
                        value=curveness,
                        iteration=self.step,
                        group_name="Curveness",
                    )
                    if self.cfg.is_vis:
                        out_path = os.path.join(self.cfg.output_dir, f"step{self.step}")
                        os.makedirs(out_path, exist_ok=True)
                        vis_during_train(
                            model=self.model,
                            dynamic=self.dynamic,
                            cfg=self.cfg,
                            out_path=out_path,
                        )

                        self.train_platform.report_video_list(
                            name="generated_motions",
                            mp4_root_path=out_path,
                            iteration=self.step,
                            group_name="generated",
                        )

                    self.evaluate()
                    self.model.train()

                self.step += 1
            if not (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        print("evaluation....")
        if not self.cfg.training.eval_during_training:
            print("Skipping evaluation during training")
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print("Running evaluation loop: [Should take about 90 min]")
            log_file = os.path.join(
                self.save_dir, f"eval_humanml_{(self.step + self.resume_step):09d}.log"
            )
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper,
                self.eval_gt_data,
                self.eval_data,
                log_file,
                replication_times=self.cfg.training.eval_rep_times,
                diversity_times=diversity_times,
                mm_num_times=mm_num_times,
                run_mm=False,
            )
            if is_rank_zero():
                wandb.log(eval_dict, step=self.step)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith("R_precision"):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(
                            name=f"top{i + 1}_" + k,
                            value=v[i],
                            iteration=self.step + self.resume_step,
                            group_name="Eval",
                        )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=v,
                        iteration=self.step + self.resume_step,
                        group_name="Eval",
                    )

        elif self.dataset in ["humanact12", "uestc"]:
            eval_args = SimpleNamespace(
                num_seeds=self.cfg.training.eval_rep_times,
                num_samples=self.cfg.training.eval_num_samples,
                batch_size=self.cfg.training.eval_batch_size,
                device=self.device,
                guidance_param=1,
                dataset=self.dataset,
                unconstrained=self.cfg.model.unconstrained,
                model_path=os.path.join(self.save_dir, self.ckpt_file_name()),
            )
            eval_dict = eval_humanact12_uestc.evaluate(
                eval_args,
                model=self.model,
                diffusion=self.dynamic,
                data=self.dataloader_train.dataset,
                cfg=self.cfg,
            )
            if is_rank_zero():
                wandb.log(eval_dict, step=self.step)
            print(
                f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}'
            )
            for k, v in eval_dict["feats"].items():
                if "unconstrained" not in k:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval",
                    )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval Unconstrained",
                    )

        end_eval = time.time()
        print(f"Evaluation time: {round(end_eval-start_eval)/60}min")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, len(batch), self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= len(batch)
            weights = 1.0

            compute_losses = functools.partial(
                self.dynamic.training_losses,
                self.ddp_model,
                micro,
                t=None,
                model_kwargs=micro_cond,
                dataset=self.dataloader_train.dataset,
            )

            if last_batch or not self.use_ddp:
                forward_dict = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    forward_dict = compute_losses()

            loss = (forward_dict["loss"] * weights).mean()
            data_range_dict = dict()
            # [bs,njoints, nfeats, nframes]
            data_range_dict["data_range_j0/data_mean"] = micro[:, 0].mean()
            data_range_dict["data_range_j0/data_std"] = micro[:, 0].std()
            data_range_dict["data_range_j0/data_min"] = micro[:, 0].min()
            data_range_dict["data_range_j0/data_max"] = micro[:, 0].max()
            if is_rank_zero():
                wandb.log(data_range_dict, step=self.step)

            log_loss_dict({k: v * weights for k, v in forward_dict.items()})
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
