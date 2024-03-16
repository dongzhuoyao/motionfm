import wandb

from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import (
    get_mdm_loader,
)  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import (
    create_model_and_diffusion,
    create_model_and_flow,
    load_model_wo_clip,
)

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel

torch.multiprocessing.set_sharing_strategy("file_system")
from tqdm import tqdm

import hydra


@hydra.main(config_path="config", config_name="config_base", version_base=None)
def main(cfg):
    if cfg.is_debug:
        if False:
            cfg.dataset = "kit"
            cfg.dynamic = "flow"
            cfg.model.text_emebed = "t5-large"
            cfg.guidance_param = 2.5
            cfg.model_path = "./outputs/kit_enc_t5large_unflatten_4gpu_occupy/12-09-2023/09-21-46/model000200056.pt"
            cfg.eval_mode = "mm_short"
        elif False:
            cfg.dataset = "kit"
            cfg.dynamic = "diffusion"
            cfg.guidance_param = 2.5
            cfg.model_path = "./pretrained/kit_trans_enc_512/model000400000.pt"
            cfg.eval_mode = "mm_short"
        print("debug mode oppen")
    fixseed(cfg.seed)
    cfg.batch_size = 32  # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(cfg.model_path))
    niter = os.path.basename(cfg.model_path).replace("model", "").replace(".pt", "")
    log_file_name = os.path.join(
        os.path.dirname(cfg.model_path), "eval_humanml_{}_{}".format(name, niter)
    )
    if cfg.guidance_param != 1.0:
        log_file_name += f"_gscale{cfg.guidance_param}"
    log_file_name += f"_{cfg.eval_mode}"
    if cfg.dynamic == "diffusion":
        log_file_name += (
            f"_diffusion_ddim{int(cfg.use_ddim)}stepnum{cfg.diffusion_steps_sample}"
        )
    elif cfg.dynamic == "flow":
        log_file_name += (
            f"_flow_{cfg.ode_kwargs['method']}step{cfg.ode_kwargs['step_size']}"
        )
    else:
        raise ValueError()
    log_file_name += f"_{datetime.now().strftime('%m-%d-%H-%M')}"
    log_file_name += ".log"

    print(f"Will save to log file [{log_file_name}]")

    print(f"Eval mode [{cfg.eval_mode}]")
    if cfg.eval_mode == "debug":
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5  # about 3 Hrs
    elif cfg.eval_mode == "wo_mm":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20  # about 12 Hrs
    elif cfg.eval_mode == "mm_short":
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating data loader...")
    SPLIT = "test"
    gt_loader = get_dataset_loader(
        name=cfg.dataset,
        batch_size=cfg.batch_size,
        num_frames=None,
        split=SPLIT,
        hml_mode="gt",
    )
    gen_loader = get_dataset_loader(
        name=cfg.dataset,
        batch_size=cfg.batch_size,
        num_frames=None,
        split=SPLIT,
        hml_mode="eval",
    )

    num_actions = gen_loader.dataset.num_actions

    if cfg.dynamic == "diffusion":
        model, dynamic = create_model_and_diffusion(cfg, gen_loader)
    elif cfg.dynamic == "flow":
        model, dynamic = create_model_and_flow(cfg, gen_loader)
    else:
        raise ValueError()

    logger.log(f"Loading checkpoints from [{cfg.model_path}]...")
    state_dict = torch.load(cfg.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if cfg.guidance_param != 1:
        # wrapping model with the classifier-free sampler
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    eval_motion_loaders = {
        "vald": lambda: get_mdm_loader(
            model,
            dynamic,
            cfg.batch_size,
            gen_loader,
            mm_num_samples,
            mm_num_repeats,
            gt_loader.dataset.opt.max_motion_length,
            num_samples_limit,
            cfg.guidance_param,
            ode_kwargs=cfg.ode_kwargs,
            cfg=cfg,
        )
    }
    eval_wrapper = EvaluatorMDMWrapper(cfg.dataset, dist_util.dev())
    evaluation(
        eval_wrapper,
        gt_loader,
        eval_motion_loaders,
        log_file_name,
        replication_times,
        diversity_times,
        mm_num_times,
        run_mm=run_mm,
    )


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print("========== Evaluating matching_score ==========")
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
                )
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f"---> [{motion_loader_name}] matching_score: {matching_score:.4f}")
        print(
            f"---> [{motion_loader_name}] matching_score: {matching_score:.4f}",
            file=file,
            flush=True,
        )

        line = f"---> [{motion_loader_name}] R_precision: "
        for i in range(len(R_precision)):
            line += "(top %d): %.4f " % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print("========== Evaluating FID ==========")
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f"---> [{model_name}] FID: {fid:.4f}")
        print(f"---> [{model_name}] FID: {fid:.4f}", file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating Diversity ==========")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f"---> [{model_name}] Diversity: {diversity:.4f}")
        print(f"---> [{model_name}] Diversity: {diversity:.4f}", file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating MultiModality ==========")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0]
                )
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        print(
            f"---> [{model_name}] Multimodality: {multimodality:.4f}",
            file=file,
            flush=True,
        )
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(
    eval_wrapper,
    gt_loader,
    eval_motion_loaders,
    log_file_name,
    replication_times,
    diversity_times,
    mm_num_times,
    run_mm=False,
):
    with open(log_file_name, "w") as f:
        all_metrics = OrderedDict(
            {
                "matching_score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "FID": OrderedDict({}),
                "Diversity": OrderedDict({}),
                "MultiModality": OrderedDict({}),
            }
        )
        print("replication_times:", replication_times)
        for replication in tqdm(
            range(replication_times), desc="replication", total=replication_times
        ):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders["ground truth"] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(
                f"==================== Replication {replication} ===================="
            )
            print(
                f"==================== Replication {replication} ====================",
                file=f,
                flush=True,
            )
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                eval_wrapper, motion_loaders, f
            )

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f"Time: {datetime.now()}")
                print(f"Time: {datetime.now()}", file=f, flush=True)
                mm_score_dict = evaluate_multimodality(
                    eval_wrapper, mm_motion_loaders, f, mm_num_times
                )

            print(f"!!! DONE !!!")
            print(f"!!! DONE !!!", file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics["matching_score"]:
                    all_metrics["matching_score"][key] = [item]
                else:
                    all_metrics["matching_score"][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics["R_precision"]:
                    all_metrics["R_precision"][key] = [item]
                else:
                    all_metrics["R_precision"][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics["FID"]:
                    all_metrics["FID"][key] = [item]
                else:
                    all_metrics["FID"][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics["Diversity"]:
                    all_metrics["Diversity"][key] = [item]
                else:
                    all_metrics["Diversity"][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics["MultiModality"]:
                        all_metrics["MultiModality"][key] = [item]
                    else:
                        all_metrics["MultiModality"][key] += [item]

        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print("========== %s Summary ==========" % metric_name)
            print("========== %s Summary ==========" % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times
                )
                mean_dict[metric_name + "_" + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}"
                    )
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}",
                        file=f,
                        flush=True,
                    )
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == "__main__":
    main()
