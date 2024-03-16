from flow.flow_matching_class import FlowMatching
from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.mdm_flow import MDM_Flow


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])


def create_model_and_diffusion(cfg, data):
    print("creating model and diffusion...")
    model = MDM(**get_model_args(cfg, data))
    dynamic = create_gaussian_diffusion(cfg)
    return model, dynamic


def create_model_and_flow(cfg, data):
    print("creating model and flow...")
    model = MDM_Flow(**get_model_args(cfg, data))
    dynamic = create_flow(cfg)

    if False:
        from accelerate import Accelerator

        _acc = Accelerator()
        model = _acc.prepare_model(model)
        assert model is not None

    return model, dynamic


def get_cond_mode(cfg):
    if cfg.model.unconstrained:
        cond_mode = "no_cond"
    elif cfg.dataset in ["kit", "humanml"]:
        cond_mode = "text"
    else:
        cond_mode = "action"
    print("cond_mode: ", cond_mode)
    return cond_mode


def get_model_args(cfg, data, clip_version="ViT-B/32", action_emb="tensor"):
    cond_mode = get_cond_mode(cfg)
    if hasattr(data.dataset, "num_actions"):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    assert cfg.model.text_emebed in ["clip", "t5-large"]

    if cfg.dataset == "humanml":
        data_rep = "hml_vec"
        njoints = 263
        nfeats = 1
    elif cfg.dataset == "kit":
        data_rep = "hml_vec"
        njoints = 251
        nfeats = 1
    else:
        # SMPL defaults
        data_rep = "rot6d"
        njoints = 25
        nfeats = 6

    return {
        "modeltype": "",
        "njoints": njoints,
        "nfeats": nfeats,
        "num_actions": num_actions,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "latent_dim": cfg.model.latent_dim,
        "ff_size": cfg.model.ff_size,
        "num_layers": cfg.model.layers,
        "num_heads": cfg.model.num_heads,
        "dropout": 0.1,
        "activation": "gelu",
        "data_rep": data_rep,
        "cond_mode": cond_mode,
        "cond_mask_prob": cfg.model.cond_mask_prob,
        "action_emb": action_emb,
        "arch": cfg.model.arch,
        "emb_trans_dec": cfg.model.emb_trans_dec,
        "clip_version": clip_version,
        "text_embed": cfg.model.text_emebed,
        "dataset": cfg.dataset,
    }


def create_gaussian_diffusion(cfg):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.0  # no scaling
    timestep_respacing = ""  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(cfg.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=cfg.model.lambda_vel,
        lambda_rcxyz=cfg.model.lambda_rcxyz,
        lambda_fc=cfg.model.lambda_fc,
    )


def create_flow(args):
    return FlowMatching(
        lambda_vel=args.model.lambda_vel,
        lambda_rcxyz=args.model.lambda_rcxyz,
        lambda_fc=args.model.lambda_fc,
    )
