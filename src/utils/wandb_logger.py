import dataclasses

import wandb

from utils.file_logger import FileLogger
from utils.logger import LogMetadata


class WandbLogger(FileLogger):
    """
    A logger that logs both locally and to wandb.
    """
    def __init__(self, metadata: LogMetadata, keys: set[str], stds: set[str], log_dir='logs', model_dir='models'):
        super().__init__(metadata, keys, stds, log_dir, model_dir)
        wandb.init(project='rl-baselines', config=dataclasses.asdict(metadata))

    def log_metadata(self):
        super().log_metadata()

    def log(self):
        self.check_data_integrity()
        self.aggregate()
        wandb.log(self.state)
        super().log()
        self.state = {}

    def save_model(self, model, name):
        super().save_model(model, name)
        wandb.save(f'{self.model_path}/{name}.pth')

    def finish(self):
        super().finish()
        wandb.finish()
