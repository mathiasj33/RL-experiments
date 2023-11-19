import wandb

from utils.file_logger import FileLogger
from utils.logger import LogMetadata


class WandbLogger(FileLogger):
    """
    A logger that logs both locally and to wandb.
    """
    def __init__(self, metadata: LogMetadata, keys: set[str], stds: set[str], log_dir='logs', model_dir='models'):
        super().__init__(metadata, keys, stds, log_dir, model_dir)
        wandb.init(project='rl-baselines', config=metadata.config, group=metadata.experiment, tags=['vpg', metadata.config.env_name])

    def log_metadata(self):
        super().log_metadata()

    def log(self):
        self.check_data_integrity()
        self.aggregate()
        wandb.log(self.state)
        super().log()
        self.state = {}

    def save_model(self, model):
        super().save_model(model)
        wandb.save(self.model_file)

    def finish(self):
        super().finish()
        wandb.finish()
