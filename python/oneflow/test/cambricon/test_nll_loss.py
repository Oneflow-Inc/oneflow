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
import random
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_nll_loss_forward(test_case, shape, reduction, device, dtype, has_weight):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device), dtype=dtype)
    C = shape[1]

    target_dims = [shape[0]] + list(shape[2:])
    target = flow.randint(0, C, target_dims, dtype=flow.int).to(flow.device(device))

    weight = None
    weight_cpu = None
    if has_weight:
        weight = flow.randn(C, dtype=dtype).to(flow.device(device))
        weight_cpu = weight.cpu()
        if dtype == flow.float16:
            weight_cpu = weight_cpu.float()

    nll = flow.nn.NLLLoss(weight=weight, reduction=reduction)
    mlu_out = nll(x, target)

    if dtype == flow.float16:
        nll = flow.nn.NLLLoss(weight=weight_cpu, reduction=reduction)
        cpu_out = nll(x.cpu().float(), target.cpu())
        test_case.assertTrue(
            np.allclose(cpu_out.numpy(), mlu_out.numpy(), 0.001, 0.001)
        )
    else:
        nll = flow.nn.NLLLoss(weight=weight_cpu, reduction=reduction)
        cpu_out = nll(x.cpu(), target.cpu())
        test_case.assertTrue(
            np.allclose(cpu_out.numpy(), mlu_out.numpy(), 0.0001, 0.0001)
        )


def _test_nll_loss_backward(test_case, shape, reduction, device, dtype, has_weight):
    x = flow.tensor(
        np.random.randn(*shape), device=flow.device(device), dtype=dtype
    ).requires_grad_(True)
    C = shape[1]

    target_dims = [shape[0]] + list(shape[2:])
    target = flow.randint(0, C, target_dims, dtype=flow.int).to(flow.device(device))

    weight = None
    weight_cpu = None
    if has_weight:
        weight = flow.randn(C, dtype=dtype).to(flow.device(device))
        weight_cpu = weight.cpu()
        if dtype == flow.float16:
            weight_cpu = weight_cpu.float()

    # mlu backward
    nll = flow.nn.NLLLoss(weight=weight, reduction=reduction)
    mlu_out = nll(x, target)
    y_grad = flow.tensor(
        np.random.randn(*mlu_out.shape), device=flow.device(device), dtype=dtype
    ).requires_grad_(True)
    mlu_dx = flow.autograd.grad(
        outputs=mlu_out,
        inputs=x,
        grad_outputs=y_grad,
        create_graph=True,
        retain_graph=True,
    )[0]

    # cpu backward
    cpu_x = flow.tensor(x.cpu().numpy(), dtype=dtype).requires_grad_(True)
    nll = flow.nn.NLLLoss(weight=weight_cpu, reduction=reduction)
    cpu_out = nll(cpu_x.float(), target.cpu())
    cpu_dx = flow.autograd.grad(
        outputs=cpu_out,
        inputs=cpu_x,
        grad_outputs=y_grad.cpu(),
        create_graph=True,
        retain_graph=True,
    )[0]

    test_case.assertTrue(
        np.allclose(cpu_dx.detach().numpy(), mlu_dx.detach().numpy(), 0.0001, 0.0001)
    )


@flow.unittest.skip_unless_1n1d()
class TestNLLLossCambriconModule(flow.unittest.TestCase):
    def test_nll_loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_nll_loss_forward,
        ]
        arg_dict["shape"] = [
            (16, 32,),
            (8, 12, 24),
        ]
        arg_dict["reduction"] = [
            "mean",
            "sum",
            "none",
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            # flow.float16,
        ]
        arg_dict["has_weight"] = [True, False]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_nll_loss_backward(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_nll_loss_backward,
        ]
        arg_dict["shape"] = [
            (16, 32,),
            (8, 12, 24),
        ]
        # TODO: mean is not supported since broadcast_div_grad is not implemented
        arg_dict["reduction"] = [
            # "mean",
            "sum",
            "none",
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float32,
            flow.float16,
        ]
        arg_dict["has_weight"] = [True, False]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
