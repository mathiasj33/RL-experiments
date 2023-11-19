import os
import shutil
import unittest

import numpy as np
import pandas as pd
import torch

from utils.file_logger import FileLogger
from utils.logger import LogMetadata


class FileLoggerTest(unittest.TestCase):

    def setUp(self) -> None:
        if not os.path.exists('test-logs'):
            os.makedirs('test-logs')
        if not os.path.exists('test-models'):
            os.makedirs('test-models')
        self.logger = FileLogger(LogMetadata(
            algorithm='vpg',
            env='cartpole',
            experiment='default',
            seed=1,
            config=dict(lr=0.01, sizes=[10, 10])
        ), keys={'Epoch', 'EpochLength', 'Return', 'Loss'}, stds={'Return', 'Loss'}, log_dir='test-logs',
            model_dir='test-models')

    def tearDown(self) -> None:
        if os.path.exists('test-logs'):
            shutil.rmtree('test-logs')
        if os.path.exists('test-models'):
            shutil.rmtree('test-models')

    def test_logging(self):
        self.logger.log_metadata()
        self.assertTrue(os.path.exists('test-logs/vpg/cartpole/default/metadata_seed_1.json'))

        self.logger.store(Epoch=1)
        self.logger.store(Return=5, EpochLength=7, Loss=4)
        self.logger.store(Return=7, EpochLength=10, Loss=3)
        self.logger.log()

        self.logger.store(Epoch=2)
        for i in range(5):
            self.logger.store(Return=i, EpochLength=i * 2, Loss=i * 3)
        self.logger.log()

        log = pd.read_csv('test-logs/vpg/cartpole/default/log_seed_1.csv')
        expected = pd.DataFrame([
            [1, 8.5, 3.5, 0.5, 6.0, 1.0],
            [2, 4.0, 6.0, np.std([i * 3 for i in range(5)]), 2.0, np.std([i for i in range(5)])]
        ], columns=['Epoch', 'EpochLength_mean', 'Loss_mean', 'Loss_std', 'Return_mean', 'Return_std'])
        pd.testing.assert_frame_equal(log, expected)

    def test_cannot_overwrite_existing_log(self):
        self.logger.store(Epoch=1, Return=5, EpochLength=7, Loss=4)
        self.logger.log()
        self.assertRaises(ValueError, FileLogger, LogMetadata(
            algorithm='vpg',
            env='cartpole',
            experiment='default',
            seed=1,
            config=dict(lr=0.01, sizes=[10, 10])
        ), keys=set(), stds=set(), log_dir='test-logs')

    def test_checks_data_integrity(self):
        self.logger.store(Epoch=1)
        self.logger.store(Return=5, EpochLength=7, Loss=4)
        self.logger.log()
        self.logger.store(Epoch=1)
        self.assertRaises(ValueError, self.logger.log)
        self.logger.store(Return=3, EpochLength=3, Loss=6)
        self.logger.log()
        self.logger.store(Return=1, EpochLength=1, Loss=1, Foo=3)
        self.assertRaises(ValueError, self.logger.log)

    def test_save_model(self):
        model = torch.nn.Linear(10, 10)
        torch.nn.init.kaiming_normal_(model.weight)
        self.logger.save_model(model)
        loaded = torch.nn.Linear(10, 10)
        loaded.load_state_dict(torch.load('test-models/vpg/cartpole/default/model_seed_1.pth'))
        torch.testing.assert_close(loaded.weight, model.weight)
