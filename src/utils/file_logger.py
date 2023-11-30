import csv
import fcntl
import json
import os

import torch

from utils.json_encoder import JsonEncoder
from utils.logger import Logger, LogMetadata


class FileLogger(Logger):
    """
    A simple logger that writes a CSV file.
    """

    def __init__(self, metadata: LogMetadata, keys: set[str], stds: set[str], log_dir='logs', model_dir='models'):
        super().__init__(metadata, keys, stds)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.log_path = os.path.join(self.log_dir, self.metadata.algorithm, self.metadata.env, self.metadata.experiment)
        self.model_path = os.path.join(self.model_dir, self.metadata.algorithm, self.metadata.env,
                                       self.metadata.experiment,
                                       f'seed_{self.metadata.seed}')
        self.log_file = f'{self.log_path}/log_seed_{self.metadata.seed}.csv'
        if os.path.exists(self.log_file):
            raise ValueError('Logging results already exist for selected experiment and seed!')
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        self.column_names = keys
        for key in stds:
            self.column_names.add(f'{key}_std')
        self.column_names = sorted(list(keys))
        self.log_metadata()

    def log_metadata(self):
        metadata_file = f'{self.log_path}/experiment_metadata.json'
        with open(metadata_file, 'w+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:  # metadata file exists already
                f.seek(0)
                previous_config = json.load(f)['config']
                if previous_config != self.metadata_as_json()['config']:
                    raise ValueError('Previous log with different config exists!')
            else:
                json.dump(self.metadata, f, indent=4, cls=JsonEncoder)
            fcntl.flock(f, fcntl.LOCK_UN)

    def metadata_as_json(self) -> dict:
        return json.loads(json.dumps(self.metadata, cls=JsonEncoder))

    def log(self):
        self.check_data_integrity()
        self.aggregate()
        first_log = not os.path.exists(self.log_file)
        with open(self.log_file, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.column_names)
            if first_log:
                writer.writeheader()
            writer.writerow(self.state)
        self.state = {}

    def save_model(self, model, name):
        torch.save(model.state_dict(), f'{self.model_path}/{name}.pth')

    def finish(self):
        pass  # nothing to do
