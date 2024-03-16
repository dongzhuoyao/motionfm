"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import hydra
import torch
import re

from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from eval.a2m.tools import save_metrics
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils.model_util import (
    create_model_and_diffusion,
    create_model_and_flow,
    load_model_wo_clip,
)


def evaluate(cfg1, model, diffusion, data, cfg):
    scale = None
    if cfg.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
        scale = {
            "action": torch.ones(cfg.batch_size) * cfg.guidance_param,
        }
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    folder, ckpt_name = os.path.split(cfg.model_path)
    if cfg.dataset == "humanact12":
        from eval.a2m.gru_eval import evaluate

        eval_results = evaluate(cfg, model, diffusion, data, cfg)
    elif cfg.dataset == "uestc":
        from eval.a2m.stgcn_eval import evaluate

        eval_results = evaluate(cfg, model, diffusion, data, cfg)
    else:
        raise NotImplementedError("This dataset is not supported.")

    # save results
    iter = int(re.findall("\d+", ckpt_name)[0])
    scale = 1 if scale is None else scale["action"][0].item()
    scale = str(scale).replace(".", "p")
    metricname = "evaluation_results_iter{}_samp{}_scale{}_a2m.yaml".format(
        iter, cfg.num_samples, scale
    )
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    return eval_results


@hydra.main(config_path="config", config_name="config_base", version_base=None)
def main(cfg):
    if cfg.is_debug:
        print("debug mode")
        cfg.guidance_param = 1.0
        cfg.eval_mode = "debug"
        cfg.model_path = "./outputs/occupy_a2m_humanct12_4gpu_300k_lambda0/19-09-2023/15-04-36/model000300000.pt"

    fixseed(cfg.seed)
    dist_util.setup_dist()

    print(f"Eval mode [{cfg.eval_mode}]")
    assert cfg.eval_mode in [
        "debug",
        "full",
    ], f"eval_mode {cfg.eval_mode} is not supported for dataset {cfg.dataset}"
    if cfg.eval_mode == "debug":
        cfg.num_samples = 10
        cfg.num_seeds = 2
        print("Debug mode, only 10 samples will be generated.")
    else:
        cfg.num_samples = 1000
        cfg.num_seeds = 20

    data_loader = get_dataset_loader(
        name=cfg.dataset,
        num_frames=60,
        batch_size=cfg.batch_size,
    )

    if cfg.dynamic == "diffusion":
        model, dynamic = create_model_and_diffusion(cfg, data_loader)
    elif cfg.dynamic == "flow":
        model, dynamic = create_model_and_flow(cfg, data_loader)
    else:
        raise ValueError()

    print(f"Loading checkpoints from [{cfg.model_path}]...")
    state_dict = torch.load(cfg.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    eval_results = evaluate(cfg, model, dynamic, data_loader.dataset, cfg)

    fid_to_print = {
        k: sum([float(vv) for vv in v]) / len(v)
        for k, v in eval_results["feats"].items()
        if "fid" in k and "gen" in k
    }
    print(fid_to_print)


if __name__ == "__main__":
    main()
