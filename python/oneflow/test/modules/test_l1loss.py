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
from automated_test_util import *


def _np_l1loss(np_input, np_target):
    np_l1 = np.abs(np_target - np_input)
    np_l1_sum = np.sum(np_l1)
    np_l1_mean = np.mean(np_l1)
    return {"none": np_l1, "mean": np_l1_mean, "sum": np_l1_sum}


def _np_l1loss_grad(np_input, np_target):
    elem_cnt = np_input.size
    np_grad = np.zeros_like(np_target)
    np_grad = np.sign(np_input - np_target)
    np_l1_grad_sum = np_grad
    np_l1_grad_mean = np_l1_grad_sum / elem_cnt
    return {"none": np_grad, "mean": np_l1_grad_mean, "sum": np_l1_grad_sum}


def _test_l1loss_impl(test_case, device, shape, reduction):
    x = np.random.randn(*shape).astype(np.float32)
    y = np.random.randn(*shape).astype(np.float32)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    loss = flow.nn.L1Loss(reduction)
    loss = loss.to(device)
    of_out = loss(input, target)
    np_out = _np_l1loss(x, y)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_l1loss_grad(x, y)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestL1LossModule(flow.unittest.TestCase):
    def test_l1loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_l1loss_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [
            (3, 5),
            (10, 9, 21),
            (14, 22, 9, 21),
            (3, 2, 4, 16, 5),
            (1,),
        ]
        arg_dict["reduction"] = ["none", "sum", "mean"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest()
    def test_l1loss_module_with_random_data(test_case):
        k = random(1, 6).to(int)
        dim0 = random(1, 10).to(int)
        dim1 = random(1, 10).to(int)
        dim2 = random(1, 10).to(int)
        dim3 = random(1, 10).to(int)
        dim4 = random(1, 10).to(int)
        reduction = oneof("none", "sum", "mean")
        loss = torch.nn.L1Loss(reduction=reduction | nothing())
        loss.train(random())
        device = random_device()
        loss.to(device)
        input = random_pytorch_tensor(
            ndim=k, dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4
        ).to(device)
        target = random_pytorch_tensor(
            ndim=k, dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4
        ).to(device)
        y = loss(input, target)
        return y


if __name__ == "__main__":
    unittest.main()
