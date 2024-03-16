# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.model_util import (
    create_model_and_diffusion,
    create_model_and_flow,
    load_model_wo_clip,
)
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
import hydra


_FFMPEG_PATH = "ffmpeg"


@hydra.main(config_path="config", config_name="config_base", version_base=None)
def main(cfg):
    if False:
        cfg.dataset = "humanml"
        cfg.dynamic = "diffusion"
        cfg.model_path = "./pretrained/humanml_trans_enc_512/model000475000.pt"
        cfg.input_text = "assets/prompts_method_compare.txt"
        cfg.guidance_param = 1.0
        cfg.num_samples = cfg.batch_size = 128
        cfg.diffusion_steps_sample = 400
        cfg.seed = 130
    elif True:
        cfg.dataset = "humanml"
        cfg.dynamic = "flow"
        cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000300000.pt"
        cfg.input_text = "assets/prompts_method_compare.txt"
        cfg.text_prompt = None
        cfg.guidance_param = 1.0
        cfg.num_samples = cfg.batch_size = 128
        cfg.seed = 130
    elif False:
        cfg.dataset = "humanml"
        cfg.dynamic = "flow"
        cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000300000.pt"
        # cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000200000.pt"
        # cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000100000.pt"
        # cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000050000.pt"
        # cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000000000.pt"
        cfg.input_text = "assets/prompts_method_compare.txt"
        cfg.text_prompt = None
        cfg.guidance_param = 1.0
        cfg.num_samples = cfg.batch_size = 128
        cfg.seed = 130
        cfg.ode_kwargs.method = "euler_raw"
        cfg.ode_kwargs.return_x_est = True
        cfg.ode_kwargs.return_x_est_num = 10
        # cfg.ode_kwargs.step_size = 0.01
    elif True:
        cfg.dataset = "humanml"
        cfg.dynamic = "flow"
        cfg.model_path = "./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000300000.pt"
        cfg.input_text = None
        cfg.text_prompt = None
        cfg.guidance_param = 2.5
        cfg.num_samples = cfg.batch_size = 128
        cfg.seed = 130
        # cfg.ode_kwargs.method = "euler_raw"
        cfg.ode_kwargs.method = "dopri5"
        cfg.ode_kwargs.return_x_est = False
        cfg.ode_kwargs.return_x_est_num = None
        cfg.ode_kwargs.step_size = 0.01
    else:
        # TODO, action2pose
        raise NotImplementedError
        cfg.dataset = "humanml"
        cfg.model_path = "./pretrained/humanml_trans_enc_512/model000475000.pt"
        cfg.input_text = "./assets/example_text_prompts.txt"
        cfg.guidance_param = 1.0

    fixseed(cfg.seed)
    out_path = cfg.output_dir
    name = os.path.basename(os.path.dirname(cfg.model_path))
    niter = os.path.basename(cfg.model_path).replace("model", "").replace(".pt", "")
    max_frames = 196 if cfg.dataset in ["kit", "humanml"] else 60
    fps = 12.5 if cfg.dataset == "kit" else 20
    n_frames = min(max_frames, int(cfg.motion_length * fps))
    print("dataset", cfg.dataset)
    print("cfg.motion_length", cfg.motion_length)
    print("max_frames", max_frames)
    print("n_frames", n_frames)
    print("fps", fps)
    is_using_data = not any(
        [cfg.input_text, cfg.text_prompt, cfg.action_file, cfg.action_name]
    )
    if is_using_data:
        print("******** Using data from the test set")
    else:
        print("******** Using data from the command line")
    dist_util.setup_dist()
    if out_path == "":
        out_path = os.path.join(
            os.path.dirname(cfg.model_path),
            "samples_{}_{}_seed{}".format(name, niter, cfg.seed),
        )
        if cfg.text_prompt != "":
            out_path += "_" + cfg.text_prompt.replace(" ", "_").replace(".", "")
        elif cfg.input_text != "":
            out_path += "_" + os.path.basename(cfg.input_text).replace(
                ".txt", ""
            ).replace(" ", "_").replace(".", "")

    # this block must be called BEFORE the dataset is loaded
    if cfg.text_prompt is not None and cfg.text_prompt != "":
        texts = [cfg.text_prompt]
        cfg.num_samples = 1
    elif cfg.input_text is not None and cfg.input_text != "":
        assert os.path.exists(cfg.input_text)
        with open(cfg.input_text, "r") as fr:
            texts = fr.readlines()
        texts = [s.replace("\n", "") for s in texts]
        cfg.num_samples = len(texts)
    elif cfg.action_name is not None and cfg.action_name:
        action_text = [cfg.action_name]
        cfg.num_samples = 1
    elif cfg.action_file is not None and cfg.action_file != "":
        assert os.path.exists(cfg.action_file)
        with open(cfg.action_file, "r") as fr:
            action_text = fr.readlines()
        action_text = [s.replace("\n", "") for s in action_text]
        cfg.num_samples = len(action_text)

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

    print("Loading dataset...")
    data_loader = load_dataset(cfg, max_frames, n_frames)
    total_num_samples = cfg.num_samples * cfg.num_repetitions

    if cfg.dynamic == "diffusion":
        model, dynamic = create_model_and_diffusion(cfg, data_loader)
    elif cfg.dynamic == "flow":
        model, dynamic = create_model_and_flow(cfg, data_loader)
    else:
        raise NotImplementedError
    print(f"Loading checkpoints from [{cfg.model_path}]...")
    state_dict = torch.load(cfg.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if cfg.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data_loader)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [
            {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
        ] * cfg.num_samples
        is_t2m = any([cfg.input_text, cfg.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [
                dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
            ]
        else:
            # a2m
            action = data_loader.dataset.action_name_to_action(action_text)
            collate_args = [
                dict(arg, action=one_action, action_text=one_action_text)
                for arg, one_action, one_action_text in zip(
                    collate_args, action, action_text
                )
            ]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_motions_est = []
    all_lengths = []
    all_text = []

    for rep_i in range(cfg.num_repetitions):
        print(f"### Sampling [repetitions #{rep_i}]")

        # add CFG scale to batch
        if cfg.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(cfg.batch_size, device=dist_util.dev()) * cfg.guidance_param
            )

        sample_fn = dynamic.p_sample_loop

        sample = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (cfg.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            sample_steps=cfg.diffusion_steps_sample,
            ode_kwargs=cfg.ode_kwargs,
        )

        is_return_est = isinstance(sample, tuple)
        if is_return_est:
            sample, x_ests = sample  # [B,T,C],[4xB,T,C]

        def _sample_to_xyz(_sample):
            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == "hml_vec":
                n_joints = 22 if _sample.shape[1] == 263 else 21
                _sample = data_loader.dataset.t2m_dataset.inv_transform(
                    _sample.cpu().permute(0, 2, 3, 1)
                ).float()
                _sample = recover_from_ric(_sample, n_joints)
                _sample = _sample.view(-1, *_sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = (
                "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
            )
            rot2xyz_mask = (
                None
                if rot2xyz_pose_rep == "xyz"
                else model_kwargs["y"]["mask"].reshape(cfg.batch_size, n_frames).bool()
            )
            _sample = model.rot2xyz(
                x=_sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )
            return _sample

        sample = _sample_to_xyz(sample)
        if is_return_est:
            x_ests = _sample_to_xyz(x_ests)

        if cfg.model.unconstrained:
            all_text += ["unconstrained"] * cfg.num_samples
        else:
            text_key = "text" if "text" in model_kwargs["y"] else "action_text"
            all_text += model_kwargs["y"][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
        if is_return_est:
            all_motions_est.append(x_ests.cpu().numpy())

        print(f"created {len(all_motions) * cfg.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    if is_return_est:
        all_motions_est = np.concatenate(all_motions_est, axis=0)
    if True:
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
        all_text = all_text[:total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    dict_2_save = {
        "motion": all_motions,
        "text": all_text,
        "lengths": all_lengths,
        "num_samples": cfg.num_samples,
        "num_repetitions": cfg.num_repetitions,
    }
    if is_return_est:
        dict_2_save["motion_est"] = all_motions_est
        dict_2_save["motion_est_num"] = cfg.ode_kwargs.return_x_est_num
    np.save(
        npy_path,
        dict_2_save,
    )
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = (
        paramUtil.kit_kinematic_chain
        if cfg.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )

    sample_files = []
    num_samples_in_out_file = 7

    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(cfg.model.unconstrained)

    for sample_i in range(cfg.num_samples):
        rep_files = []
        for rep_i in range(cfg.num_repetitions):
            caption = all_text[rep_i * cfg.batch_size + sample_i]
            length = all_lengths[rep_i * cfg.batch_size + sample_i]
            motion = all_motions[rep_i * cfg.batch_size + sample_i].transpose(2, 0, 1)[
                :length
            ]
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(
                animation_save_path,
                skeleton,
                motion,
                dataset=cfg.dataset,
                title=caption,
                fps=fps,
            )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(
            cfg,
            out_path,
            row_print_template,
            all_print_template,
            row_file_template,
            all_file_template,
            caption,
            num_samples_in_out_file,
            rep_files,
            sample_files,
            sample_i,
        )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


def save_multiple_samples(
    args,
    out_path,
    row_print_template,
    all_print_template,
    row_file_template,
    all_file_template,
    caption,
    num_samples_in_out_file,
    rep_files,
    sample_files,
    sample_i,
    ffmpeg_path=_FFMPEG_PATH,
):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f" -i {f} " for f in rep_files]
    hstack_args = (
        f" -filter_complex hstack=inputs={args.num_repetitions}"
        if args.num_repetitions > 1
        else ""
    )
    ffmpeg_rep_cmd = (
        f"{ffmpeg_path} -y -loglevel warning "
        + "".join(ffmpeg_rep_files)
        + f"{hstack_args} {all_rep_save_path}"
    )
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (
        sample_i + 1
    ) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(
            sample_i - len(sample_files) + 1, sample_i
        )
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(
            all_print_template.format(
                sample_i - len(sample_files) + 1, sample_i, all_sample_save_file
            )
        )
        ffmpeg_rep_files = [f" -i {f} " for f in sample_files]
        vstack_args = (
            f" -filter_complex vstack=inputs={len(sample_files)}"
            if len(sample_files) > 1
            else ""
        )
        ffmpeg_rep_cmd = (
            f"{ffmpeg_path} -y -loglevel warning "
            + "".join(ffmpeg_rep_files)
            + f"{vstack_args} {all_sample_save_path}"
        )
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = "sample{:02d}.mp4"
    all_file_template = "samples_{:02d}_to_{:02d}.mp4"
    if unconstrained:
        sample_file_template = "row{:02d}_col{:02d}.mp4"
        sample_print_template = "[{} row #{:02d} column #{:02d} | -> {}]"
        row_file_template = row_file_template.replace("sample", "row")
        row_print_template = "[{} row #{:02d} | all columns | -> {}]"
        all_file_template = all_file_template.replace("samples", "rows")
        all_print_template = "[rows {:02d} to {:02d} | -> {}]"
    else:
        sample_file_template = "sample{:02d}_rep{:02d}.mp4"
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = "[samples {:02d} to {:02d} | all repetitions | -> {}]"

    return (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    )


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
    )
    if args.dataset in ["kit", "humanml"]:
        data.dataset.t2m_dataset.fixed_length = n_frames
        # what??
    return data


if __name__ == "__main__":
    main()
