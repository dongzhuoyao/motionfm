import os
import wandb

from utils.dist_util import is_rank_zero


class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task

        path, name = os.path.split(save_dir)
        self.task = Task.init(
            project_name="motion_diffusion", task_name=name, output_uri=path
        )
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(
            title=group_name, series=name, iteration=iteration, value=value
        )

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f"{group_name}/{name}", value, iteration)

    def close(self):
        self.writer.close()


class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir):
        wandb_name = save_dir
        wandb.init(project="motionfm", config=None, name=wandb_name)

    def report_scalar(self, name, value, iteration, group_name=None):
        wandb.log(f"{group_name}/{name}", value, iteration)

    def close(self):
        pass


class Wandb_ClearML_Platform(TrainPlatform):
    def __init__(self, save_dir, wandb_cfg, cfg, open_cleanml=False):
        self.open_cleanml = open_cleanml
        if is_rank_zero():
            project_name = wandb_cfg.project
            wandb.finish()
            wandb.init(
                **wandb_cfg,
                settings=wandb.Settings(
                    start_method="fork", _disable_stats=True, _disable_meta=True
                ),
                config=dict(cfg),
            )

            ######
            if open_cleanml:
                from clearml import Task

                path, name = os.path.split(save_dir)
                self.task = Task.init(
                    project_name=project_name, task_name=name, output_uri=path
                )
                self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name=None):
        if is_rank_zero():
            wandb_dict = dict()
            wandb_dict[f"{group_name}/{name}"] = value
            wandb.log(wandb_dict, step=iteration)
            ######
            if self.open_cleanml:
                self.logger.report_scalar(
                    title=group_name, series=name, iteration=iteration, value=value
                )

    def report_video(self, name, local_path, iteration, group_name=None):
        if is_rank_zero():
            wandb_dict = dict()
            wandb_dict[f"{group_name}/{name}"] = wandb.Video(
                local_path, fps=4, format="gif"
            )
            wandb.log(wandb_dict, step=iteration)
            ######
            if self.open_cleanml:
                self.logger.report_media(
                    title=group_name,
                    series=name,
                    iteration=iteration,
                    local_path=local_path,
                )

    def report_video_list(self, name, mp4_root_path, iteration, group_name=None):
        if is_rank_zero():
            local_path_list = [
                os.path.join(mp4_root_path, f)
                for f in os.listdir(mp4_root_path)
                if f.endswith(".mp4") or f.endswith(".gif")
            ]
            wandb_dict = dict()
            wandb_dict[f"{group_name}/{name}"] = [
                wandb.Video(local_path, fps=4, format="gif")
                for local_path in local_path_list
            ]
            wandb.log(wandb_dict, step=iteration)
            ######
            if self.open_cleanml:
                for local_path in local_path_list:
                    self.logger.report_media(
                        title=group_name,
                        series=name,
                        iteration=iteration,
                        local_path=local_path,
                    )

    def report_args(self, args, name):
        if self.open_cleanml:
            if is_rank_zero():
                self.task.connect(args, name=name)
                if False:
                    config = (
                        omegaconf.OmegaConf.to_container(
                            cfg,
                            resolve=True,
                            throw_on_missing=False,
                        ),
                    )

    def close(self):
        if self.open_cleanml:
            if is_rank_zero():
                self.task.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass
