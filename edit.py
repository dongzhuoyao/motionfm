# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from einops import rearrange
import hydra
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.model_util import (
    create_model_and_diffusion,
    load_model_wo_clip,
    create_model_and_flow,
)
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil


@hydra.main(config_path="config", config_name="config_base", version_base=None)
def main(cfg):
    if cfg.is_debug:
        if False:
            cfg.dataset = "kit"  # "humanml"
            # cfg.model_path = "./pretrained/humanml_trans_enc_512/model000200000.pt"
            cfg.model_path = "./pretrained/kit_trans_enc_512/model000400000.pt"
            cfg.dynamic = "diffusion"
            cfg.edit_mode = "in_between"
            cfg.guidance_param = 1.0
            cfg.text_condition = (
                "the person walked forward and is picking up his toolbox."
            )
        elif True:
            cfg.dataset = "humanml"
            cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000300000.pt"
            # cfg.model_path = (
            #   "./outputs/kit_trans_enc_512_4gpu/07-09-2023/17-49-00/model000200000.pt"
            # )
            cfg.dynamic = "flow"

            cfg.guidance_param = 1.0
            # cfg.text_condition = ("the person walked forward and is picking up his toolbox.")
            # cfg.text_condition = "a person is stretching their arms."
            # cfg.text_condition = "a person doing jumping jacks."
            # cfg.text_condition = "A person got down and is crawling across the floor."
            # cfg.text_condition = "a man walks in a curved line."
            # cfg.text_condition = "a person is walking in a straight line."
            # cfg.text_condition = "he throws something very high."
            use_random_text = True  # a useful flag

            if False:
                cfg.edit_mode = "in_between"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.edit_till = 1.0
                cfg.ode_kwargs.step_size = 0.01
                cfg.edit_alter_prompts = None
            elif False:  # need x_est
                cfg.edit_mode = "in_between"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.edit_till = 1.0
                cfg.ode_kwargs.step_size = 0.01
                cfg.edit_alter_prompts = None
                cfg.ode_kwargs.return_x_est = True
                cfg.ode_kwargs.return_x_est_num = 10
            elif True:  ## wenzhe, in_between,  # tell replacement, and alter_prompts
                cfg.edit_mode = "in_between"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.edit_till = 1.0
                cfg.ode_kwargs.step_size = 0.01
                cfg.edit_alter_prompts = "assets/alter_prompts_inbetween.txt"
            elif False:
                cfg.edit_mode = "upper_body"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.edit_till = 0.1
                cfg.ode_kwargs.step_size = 0.01
            elif False:  # wenzhe, prediction
                cfg.edit_mode = "prediction"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.edit_till = 0.5
                cfg.ode_kwargs.step_size = 0.01
            elif False:
                cfg.edit_mode = "prediction"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.edit_till = 1.0
                cfg.ode_kwargs.step_size = 0.01
                cfg.ode_kwargs.return_x_est = True
                cfg.ode_kwargs.return_x_est_num = 8

            elif False:  # done's work well now
                cfg.edit_mode = "interpolate"
                cfg.ode_kwargs.method = "euler_replacement_edit_till"
                cfg.ode_kwargs.step_size = 1

            elif False:  # done's work well now
                cfg.edit_mode = "in_between"
                cfg.ode_kwargs.method = "odenoise_euler_replacement"
                cfg.ode_kwargs.step_size = 0.01
            elif False:  # we decide don't use it it paper
                cfg.edit_mode = "variation"
                cfg.ode_kwargs.method = "variation_euler_replacement"
                cfg.ode_kwargs.step_size = 0.01
            else:
                raise NotImplementedError

        elif False:
            cfg.dataset = "humanml"
            cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000300000.pt"
            cfg.dynamic = "flow"
            cfg.edit_mode = "upper_body"  # "in_between"
            cfg.guidance_param = 1.0
            cfg.text_condition = (
                "the person walked forward and is picking up his toolbox."
            )
            cfg.ode_kwargs.method = "euler_replacement"
            # cfg.ode_kwargs.method = "euler_raw"

        print("is_debug is True, using default config for debugging")
    if use_random_text:
        cfg.num_samples = cfg.batch_size = 64
        cfg.seed = 128
    ##########################

    fixseed(cfg.seed)
    out_path = cfg.output_dir
    name = os.path.basename(os.path.dirname(cfg.model_path))
    niter = os.path.basename(cfg.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if cfg.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if cfg.dataset == "kit" else 20
    dist_util.setup_dist()
    if out_path == "":
        out_path = os.path.join(
            os.path.dirname(cfg.model_path),
            f"edit_{name}_{niter}_{cfg.edit_mode}_seed{cfg.seed}",
        )
        if cfg.text_condition != "":
            out_path += "_" + cfg.text_condition.replace(" ", "_").replace(".", "")
        print("out_path is empty, rewrite it", out_path)
    else:
        print("out_path is not empty, using it", out_path)
    out_path = out_path + f"_s{cfg.seed}"
    print("********* out_path is", out_path)

    print("Loading dataset...")
    assert (
        cfg.num_samples <= cfg.batch_size
    ), f"Please either increase batch_size({cfg.batch_size}) or reduce num_samples({cfg.num_samples})"
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    cfg.batch_size = (
        cfg.num_samples
    )  # Sampling a single batch from the testset, with exactly args.num_samples
    data_loader = get_dataset_loader(
        name=cfg.dataset,
        batch_size=cfg.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="train",
    )  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = cfg.num_samples * cfg.num_repetitions

    print("Creating model and diffusion...")
    if cfg.dynamic == "diffusion":
        model, dynamic = create_model_and_diffusion(cfg, data_loader)
    elif cfg.dynamic == "flow":
        model, dynamic = create_model_and_flow(cfg, data_loader)
    else:
        raise NotImplementedError

    print(f"Loading checkpoints from [{cfg.model_path}]...")
    state_dict = torch.load(cfg.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(
        model
    )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    iterator = iter(data_loader)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    if use_random_text:
        print("text_condition is empty, using random text from the dataloader.")
    else:
        print("using fixed text_condition:", cfg.text_condition)
        texts = [cfg.text_condition] * cfg.num_samples
        model_kwargs["y"]["text"] = texts
    if cfg.text_condition == "":
        cfg.guidance_param = 0.0  # Force unconditioned generation
        print("cfg.text_condition == " ", force generating unconditioned samples")

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs["y"]["inpainted_motion"] = input_motions
    if cfg.edit_mode in ["in_between", "variation"]:
        model_kwargs["y"]["inpainting_mask"] = torch.ones_like(
            input_motions, dtype=torch.bool, device=input_motions.device
        )  # True means use gt motion
        for i, length in enumerate(model_kwargs["y"]["lengths"].cpu().numpy()):
            start_idx, end_idx = int(cfg.prefix_end * length), int(
                cfg.suffix_start * length
            )
            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(
                range(end_idx, max_frames)
            )
            model_kwargs["y"]["inpainting_mask"][
                i, :, :, start_idx:end_idx
            ] = False  # do inpainting in those frames
    elif cfg.edit_mode in ["prediction"]:
        model_kwargs["y"]["inpainting_mask"] = torch.ones_like(
            input_motions, dtype=torch.bool, device=input_motions.device
        )  # True means use gt motion
        for i, length in enumerate(model_kwargs["y"]["lengths"].cpu().numpy()):
            start_idx = int(cfg.prefix_end * length)
            gt_frames_per_sample[i] = list(range(0, start_idx))
            model_kwargs["y"]["inpainting_mask"][
                i, :, :, start_idx:
            ] = False  # do inpainting in those (future) frames
    elif cfg.edit_mode == "interpolate":
        RARTIO = 5
        model_kwargs["y"]["inpainting_mask"] = torch.ones_like(
            input_motions, dtype=torch.bool, device=input_motions.device
        )  # True means use gt motion
        for i, length in enumerate(model_kwargs["y"]["lengths"].cpu().numpy()):
            start_idx, end_idx = int(cfg.prefix_end * length), int(
                cfg.suffix_start * length
            )
            gt_frames_per_sample[i] = [
                _i * RARTIO for _i in range(0, max_frames // RARTIO)
            ]
            _inpainted_mask = np.array(
                [_i for _i in range(max_frames) if _i not in gt_frames_per_sample[i]]
            )
            model_kwargs["y"]["inpainting_mask"][
                i, :, :, _inpainted_mask
            ] = False  # do inpainting in those frames
    elif cfg.edit_mode == "upper_body":
        model_kwargs["y"]["inpainting_mask"] = torch.tensor(
            humanml_utils.HML_LOWER_BODY_MASK,
            dtype=torch.bool,
            device=input_motions.device,
        )  # True is lower body data
        model_kwargs["y"]["inpainting_mask"] = (
            model_kwargs["y"]["inpainting_mask"]
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(
                input_motions.shape[0],
                1,
                input_motions.shape[2],
                input_motions.shape[3],
            )
        )

    all_motions = []
    all_motions_est = []
    all_lengths = []
    all_text = []

    for rep_i in range(cfg.num_repetitions):
        print(f"### Start sampling [repetitions #{rep_i}]")

        # add CFG scale to batch
        model_kwargs["y"]["scale"] = (
            torch.ones(cfg.batch_size, device=dist_util.dev()) * cfg.guidance_param
        )

        sample_fn = dynamic.p_sample_loop

        sample = sample_fn(
            model,
            (cfg.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            ode_kwargs=cfg.ode_kwargs,
        )
        is_return_x_est = isinstance(sample, tuple)

        n_joints = 22 if model.njoints == 263 else 21

        def _sample_to_xyz(sample):
            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == "hml_vec":
                sample = data_loader.dataset.t2m_dataset.inv_transform(
                    sample.cpu().permute(0, 2, 3, 1)
                ).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            return sample

        if is_return_x_est:
            sample, x_est = sample
            x_est = _sample_to_xyz(x_est)

        sample = _sample_to_xyz(sample)

        all_text += model_kwargs["y"]["text"]
        all_motions.append(sample.cpu().numpy())
        if is_return_x_est:
            x_est = rearrange(x_est, "(t b) w h c -> t b w h c", b=len(sample))
            all_motions_est.append(x_est.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

        print(f"created {len(all_motions) * cfg.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    if is_return_x_est:
        all_motions_est = np.concatenate(all_motions_est, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    input_motions = _sample_to_xyz(input_motions).cpu().numpy()
    dict_4_save = {
        "motion": all_motions,
        "motion_gt": input_motions,
        "text": all_text,
        "lengths": all_lengths,
        "num_samples": cfg.num_samples,
        "num_repetitions": cfg.num_repetitions,
    }
    if is_return_x_est:
        dict_4_save["motion_est"] = (all_motions_est,)  # [8,motion_num,22,6,seq]
    np.save(npy_path, dict_4_save)

    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    if False:
        np.save(
            npy_path.replace(".npy", "_with_gt.npy"),
            {
                "motion_gt": input_motions,
                "motion": all_motions,
                "text": all_text,
                "lengths": all_lengths,
                "num_samples": cfg.num_samples,
                "num_repetitions": cfg.num_repetitions,
            },
        )
    elif False:  # wenzhe's option
        np.save(
            npy_path.replace(".npy", "_with_gt.npy"),
            {
                "motion": input_motions,
                "text": all_text,
                "lengths": all_lengths,
                "num_samples": cfg.num_samples,
                "num_repetitions": cfg.num_repetitions,
            },
        )

    print(f"saving visualizations to [{out_path}]...")
    skeleton = (
        paramUtil.kit_kinematic_chain
        if cfg.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )

    for sample_i in range(cfg.num_samples):
        caption = "Input Motion"
        length = model_kwargs["y"]["lengths"][sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = "input_motion{:02d}.gif".format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(
            animation_save_path,
            skeleton,
            motion,
            title=caption,
            dataset=cfg.dataset,
            fps=fps,
            vis_mode="gt",
            gt_frames=gt_frames_per_sample.get(sample_i, []),
        )
        for rep_i in range(cfg.num_repetitions):
            caption = all_text[rep_i * cfg.batch_size + sample_i]
            if caption == "":
                caption = "Edit [{}] unconditioned".format(cfg.edit_mode)
            else:
                caption = "Edit [{}]: {}".format(cfg.edit_mode, caption)
            length = all_lengths[rep_i * cfg.batch_size + sample_i]
            motion = all_motions[rep_i * cfg.batch_size + sample_i].transpose(2, 0, 1)[
                :length
            ]
            save_file = "sample{:02d}_rep{:02d}.gif".format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(
                animation_save_path,
                skeleton,
                motion,
                title=caption,
                dataset=cfg.dataset,
                fps=fps,
                vis_mode=cfg.edit_mode,
                gt_frames=gt_frames_per_sample.get(sample_i, []),
            )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, "sample{:02d}.mp4".format(sample_i))
        ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
        hstack_args = f" -filter_complex hstack=inputs={cfg.num_repetitions+1}"
        ffmpeg_rep_cmd = (
            f"ffmpeg -y -loglevel warning "
            + "".join(ffmpeg_rep_files)
            + f"{hstack_args} {all_rep_save_file}"
        )
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


if __name__ == "__main__":
    main()
