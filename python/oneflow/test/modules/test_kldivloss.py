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


def _np_kldivloss(np_input, np_target, np_log_target):
    if np_log_target:
        np_kl_div_loss = np.exp(np_target) * (np_target - np_input)
    else:
        np_kl_div_out_loss = np_target * (np.log(np_target) - np_input)
        np_zeros = np.zeros_like(np_kl_div_out_loss, dtype=np.float32)
        np_kl_div_loss = np.where(np_target > 0, np_kl_div_out_loss, np_zeros)
    return {
        "none": np_kl_div_loss,
        "mean": np.mean(np_kl_div_loss),
        "sum": np.sum(np_kl_div_loss),
    }


def _np_kldivloss_grad(input, target, np_log_target):
    elem_cnt = input.size
    if np_log_target:
        _np_diff = -np.exp(target)
    else:
        _np_diff = -target
        _zero_index = np.where(target > 0, 1, 0)
        _np_diff = _np_diff * _zero_index
    return {"none": _np_diff, "mean": _np_diff / elem_cnt, "sum": _np_diff}


def _test_kldivloss_forward(test_case, device, shape, reduction, log_target):
    x = np.random.randn(*shape)
    y = np.random.randn(*shape)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    loss = flow.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    loss = loss.to(device)
    of_out = loss(input, target)
    np_out = _np_kldivloss(x, y, log_target)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_kldivloss_backward(test_case, device, shape, reduction, log_target):
    x = np.random.randn(*shape)
    y = np.random.randn(*shape)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    loss = flow.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    loss = loss.to(device)
    of_out = loss(input, target)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_kldivloss_grad(x, y, log_target)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestKLDivLossModule(flow.unittest.TestCase):
    def test_kldivloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_kldivloss_forward, _test_kldivloss_backward]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [
            (3, 5),
            (10, 9, 21),
            (14, 22, 9, 21),
            (3, 2, 4, 16, 5),
            (1,),
        ]
        arg_dict["reduction"] = ["none", "mean", "sum"]
        arg_dict["log_target"] = [False, True]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
