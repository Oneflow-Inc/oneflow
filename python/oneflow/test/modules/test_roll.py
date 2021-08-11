import unittest
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
from automated_test_util import *

import oneflow as flow
import oneflow.unittest

@flow.unittest.skip_unless_1n1d()
class TestRoll(flow.unittest.TestCase):
    @autotest()
    def test_roll_with_random_data(test_case):
        device = 'cpu'
        x = random_pytorch_tensor(ndim=2).to(device)
        shifts = random(1, 6).to(Union[Tuple[int]])
        dims = random(0, 1).to(Union[Tuple[int]]) 
        print(shifts.value())
        print(dims.value())
        return torch.roll(
                x, 
                shifts=shifts, 
                dims=dims
                )

if __name__ == "__main__":
    unittest.main()
