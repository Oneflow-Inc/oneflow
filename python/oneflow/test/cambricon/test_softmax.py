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

import oneflow as flow
import oneflow.unittest


def _test_softmax_forward(test_case, shape, device, dtype):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    mlu_out = flow.softmax(x)
    if dtype == flow.float16:
        cpu_out = flow.softmax(x.cpu().float())
        test_case.assertTrue(
            np.allclose(cpu_out.numpy(), mlu_out.numpy(), 0.001, 0.001)
        )
    else:
        cpu_out = flow.softmax(x.cpu())
        test_case.assertTrue(
            np.allclose(cpu_out.numpy(), mlu_out.numpy(), 0.0001, 0.0001)
        )


def _test_softmax_backward(test_case, shape, dtype):
    x_np = np.random.randn(*shape)
    y_grad_np = np.random.randn(*shape)

    rtol = 1e-3 if dtype == flow.float16 else 1e-4
    atol = 1e-3 if dtype == flow.float16 else 1e-4

    def _get_softmax_grad(device):
        x = flow.tensor(x_np, device=flow.device(device), dtype=dtype).requires_grad_(
            True
        )
        y_grad = flow.tensor(
            y_grad_np, device=flow.device(device), dtype=dtype
        ).requires_grad_(True)
        if device == "cpu":
            x = x.float()
            y_grad = y_grad.float()
        y = flow.softmax(x)

        dx = flow.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=y_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        return dx

    dx_cpu = _get_softmax_grad("cpu")
    dx_mlu = _get_softmax_grad("mlu")

    test_case.assertTrue(
        np.allclose(
            dx_cpu.detach().numpy(),
            dx_mlu.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestSoftmaxCambriconModule(flow.unittest.TestCase):
    def test_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_softmax_forward,
        ]
        arg_dict["shape"] = [
            (16, 32,),
            (12, 16, 24),
            (8, 12, 16, 24),
            (4, 8, 12, 16, 24),
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_softmax_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_softmax_backward,
        ]
        arg_dict["shape"] = [
            (16, 32,),
            (12, 16, 24),
            (8, 12, 16, 24),
            (4, 8, 12, 16, 24),
        ]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
