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
import oneflow.unittest


def _np_mseloss(np_input, np_target):
    np_mse = np.square(np_target - np_input)
    np_mse_mean = np.mean(np_mse)
    np_mse_sum = np.sum(np_mse)
    return {"none": np_mse, "mean": np_mse_mean, "sum": np_mse_sum}


def _np_mseloss_grad(np_input, np_target):
    elem_cnt = np_input.size
    np_mse_grad_sum = -2 * (np_target - np_input)
    np_mse_grad_mean = np_mse_grad_sum / elem_cnt
    return {"none": np_mse_grad_sum, "mean": np_mse_grad_mean, "sum": np_mse_grad_sum}


def _test_mseloss_impl(test_case, device, shape, reduction):
    x = np.random.randn(*shape)
    y = np.random.randn(*shape)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    loss = flow.nn.MSELoss(reduction=reduction)
    loss = loss.to(device)
    of_out = loss(input, target)
    np_out = _np_mseloss(x, y)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_mseloss_grad(x, y)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestMSELossModule(flow.unittest.TestCase):
    def test_mseloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_mseloss_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [
            (3, 5),
            (10, 9, 21),
            (14, 22, 9, 21),
            (3, 2, 4, 16, 5),
            (1,),
        ]
        arg_dict["reduction"] = ["none", "mean", "sum"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
