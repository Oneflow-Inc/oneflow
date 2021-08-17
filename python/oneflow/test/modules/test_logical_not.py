"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
from automated_test_util import *


def _test_logical_not(test_case, shape, device):
    np_input = np.random.randint(3, size=shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = flow.logical_not(input)
    np_out = np.logical_not(np_input)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tensor_logical_not(test_case, shape, device):
    np_input = np.random.randint(3, size=shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = input.logical_not(input)
    np_out = np.logical_not(np_input)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


@flow.unittest.skip_unless_1n1d()
class TestLogicalNotModule(flow.unittest.TestCase):

    @autotest(n=10, auto_backward=False)
    def test_logical_not_with_random_data(test_case):
        device = random_device()
        #device = 'cpu'
        shape = random_tensor().value().shape
        x = random_pytorch_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = torch.logical_not(x)
        return y


if __name__ == "__main__":
    unittest.main() 