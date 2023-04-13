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
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


def _test_view(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = input.view(2, 2, 2, -1)
    of_shape = of_out.numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))


def _test_view_flow_size(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    shape = flow.Size([2, 2, 2, -1])
    of_out = input.view(shape)
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_shape))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestView(flow.unittest.TestCase):
    # TODO:(zhaoluyang) add test case that trigger tensor.view's check
    def test_view(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_view,
            _test_view_flow_size,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, check_graph=True)
    def test_view_with_0_dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y1 = torch.reshape(x, shape=(-1,))
        y2 = x.view((1, 1, 1))
        test_case.assertTrue(x.oneflow.stride() == x.pytorch.stride())
        test_case.assertTrue(y1.oneflow.stride() == y1.pytorch.stride())
        test_case.assertTrue(y2.oneflow.stride() == y2.pytorch.stride())
        return y2


if __name__ == "__main__":
    unittest.main()
