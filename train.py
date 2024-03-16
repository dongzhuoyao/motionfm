# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
from utils.fixseed import fixseed
from utils import dist_util
from training_loop_difusion import TrainLoop_Diffusion
from training_loop_flow import TrainLoop_Flow
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, create_model_and_flow
from train_platforms import (
    ClearmlPlatform,
    Wandb_ClearML_Platform,
)  # required for the eval operation
import hydra
import torch


@hydra.main(config_path="config", config_name="config_base", version_base=None)
def main(cfg):
    # args = train_args()
    if cfg.is_debug:
        cfg.name = "debuggg"
        cfg.training.overwrite = True
        # cfg.dataset = "humanact12"  # kit
        cfg.dataset = "kit"  # kit
        cfg.training.train_platform_type = "Wandb_ClearML_Platform"
        cfg.training.num_steps = 100
        cfg.training.eval_during_training = 0
        cfg.guidance_param = 1.0
        cfg.dynamic = "flow"
        # cfg.model.text_emebed = "t5-large"
        cfg.model.text_emebed = "clip"
        cfg.model.arch = "trans_dec"
        cfg.training.eval_during_training = True
        print("is_debug: ", cfg.is_debug)
    else:
        print("is_debug: ", cfg.is_debug)

    os.makedirs(cfg.training.save_dir, exist_ok=True)

    fixseed(cfg.seed)
    dist_util.setup_dist()
    train_platform = Wandb_ClearML_Platform(cfg.training.save_dir, cfg.wandb, cfg=cfg)
    train_platform.report_args(cfg, name="Args")

    if cfg.training.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")

    elif os.path.exists(cfg.training.save_dir) and not cfg.training.overwrite:
        raise FileExistsError(
            "save_dir [{}] already exists.".format(cfg.training.save_dir)
        )
    elif not os.path.exists(cfg.training.save_dir):
        os.makedirs(cfg.training.save_dir)

    data_loader = get_dataset_loader(
        name=cfg.dataset,
        batch_size=cfg.batch_size,
        num_frames=cfg.training.num_frames,
        num_workers=cfg.num_workers,
        is_debug=cfg.is_debug,
    )

    if cfg.dynamic == "diffusion":
        model, dynamic = create_model_and_diffusion(cfg, data_loader)
    elif cfg.dynamic == "flow":
        model, dynamic = create_model_and_flow(cfg, data_loader)
    else:
        raise NotImplementedError
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()
    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)
    )
    print("Training...")

    data_shape = next(iter(data_loader))[0].shape

    fixed_noise = torch.randn(data_shape).to('cuda') #torch.randn([128, *data_shape[1:]])
    if cfg.dynamic == "diffusion":
        TrainLoop_Diffusion(
            cfg, train_platform, model, dynamic, data_loader, fixed_noise
        ).run_loop()
    elif cfg.dynamic == "flow":
        TrainLoop_Flow(
            cfg, train_platform, model, dynamic, data_loader, fixed_noise
        ).run_loop()
    else:
        raise NotImplementedError
    train_platform.close()


if __name__ == "__main__":
    main()
