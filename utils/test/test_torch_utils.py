import unittest
import torch
from torch.nn import Sequential, Linear

from utils.torch_utils import polyak_average


class TorchUtilsTest(unittest.TestCase):
    def test_polyak_average(self):
        net1 = Sequential(
            Linear(3, 2),
            Linear(2, 1)
        )
        net2 = Sequential(
            Linear(3, 2),
            Linear(2, 1)
        )
        ratio = 0.3
        w1 = ratio * net1[0].weight.data + (1 - ratio) * net2[0].weight.data
        w2 = ratio * net1[1].weight.data + (1 - ratio) * net2[1].weight.data
        polyak_average(net1, net2, ratio)

        self.assertTrue(torch.allclose(net1[0].weight.data, w1))
        self.assertTrue(torch.allclose(net1[1].weight.data, w2))


if __name__ == '__main__':
    unittest.main()
