from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class LogMetadata:
    algorithm: str
    env: str
    experiment: str
    seed: int
    config: any


class Logger(ABC):
    def __init__(self, metadata: LogMetadata, keys: set[str], stds: set[str]):
        """
        :param keys: specifies the keys for the values that this logger saves.
        :param stds: specifies for which keys both the mean and standard deviation is stored.
        """
        self.metadata = metadata
        self.keys = keys
        self.stds = stds
        self.state = {}

    @abstractmethod
    def log_metadata(self):
        """
        Saves the metadata of the logger.
        """
        pass

    @abstractmethod
    def log(self):
        """
        Saves the current state of the logger as a single row of data and resets the logger's state.
        """
        pass

    @abstractmethod
    def save_model(self, model):
        """
        Saves the trained model.
        """
        pass

    def store(self, **kwargs):
        """
        Stores the given kwargs in the logger's current state. If there are multiple values for a given key, stores all
        given values.
        """
        for key, value in kwargs.items():
            if key not in self.state:
                self.state[key] = value
            else:
                current_value = self.state[key]
                if isinstance(current_value, list):
                    current_value.append(value)
                else:
                    self.state[key] = [current_value, value]

    def check_data_integrity(self):
        if self.state.keys() != self.keys:
            raise ValueError(
                f'Specified keys do not match given keys.\nSpecified: {self.keys}\nGiven: {self.state.keys()}')

    def aggregate(self):
        """
        Aggregates values for keys if there are multiple values for a key.
        """
        for key, value in list(self.state.items()):
            if isinstance(value, list):
                self.state[f'{key}_mean'] = np.mean(value)
                if key in self.stds:
                    self.state[f'{key}_std'] = np.std(value)
                del self.state[key]
