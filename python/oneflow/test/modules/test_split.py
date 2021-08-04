import unittest
import numpy as np
from automated_test_util import *

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSplit(flow.unittest.TestCase):
    @autotest()
    def test_flow_split_with_random_data(test_case):
        k0 = random(2, 6)
        k1 = random(2, 6)
        k2 = random(2, 6)
        rand_dim = random(0, 3).to(int)
        device = random_device()
        x = random_pytorch_tensor(ndim=3, dim0=k0, dim1=k1, dim3=k2).to(device)
        return torch.split(x, split_size_or_sections=2, dim=rand_dim)

    @autotest()
    def test_flow_split_sizes_with_random_data(test_case):
        k0 = random(2, 6)
        k1 = 7
        k2 = random(2, 6)
        device = random_device()
        x = random_pytorch_tensor(ndim=3, dim0=k0, dim1=k1, dim3=k2).to(device)
        return torch.split(x, split_size_or_sections=[1, 2, 3, 1], dim=1)


if __name__ == "__main__":
    unittest.main()
