import csv
import json
import os
from dataclasses import asdict

from utils.logger import Logger, LogMetadata


class CsvLogger(Logger):
    def __init__(self, metadata: LogMetadata, keys: set[str], stds: set[str], log_dir='logs'):
        super().__init__(metadata, keys, stds, log_dir)
        self.path = os.path.join(self.log_dir, self.metadata.algorithm, self.metadata.env, self.metadata.experiment)
        self.filename = f'{self.path}/log_seed_{self.metadata.seed}.csv'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if os.path.exists(self.filename):
            raise ValueError('Logging results already exist for selected experiment and seed!')
        self.column_names = None

    def log_metadata(self):
        with open(f'{self.path}/metadata_seed_{self.metadata.seed}.json', 'w+') as f:
            json.dump(asdict(self.metadata), f, indent=4)

    def log(self):
        self.check_data_integrity()
        self.aggregate()
        first_log = False
        if self.column_names is None:
            first_log = True
            self.column_names = sorted(self.state.keys())

        with open(self.filename, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.column_names)
            if first_log:
                writer.writeheader()
            writer.writerow(self.state)
        self.state = {}
