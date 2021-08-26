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
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

import oneflow as flow
import oneflow.unittest


def _np_smoothl1loss(np_input, np_target, beta=1.0):
    original_shape = np_input.shape
    elem_cnt = np_input.size
    np_input = np_input.reshape(-1)
    np_target = np_target.reshape(-1)
    loss = np.zeros(elem_cnt).astype(np_input.dtype)
    for i in np.arange(elem_cnt):
        abs_diff = abs(np_input[i] - np_target[i])
        if abs_diff < beta:
            loss[i] = 0.5 * abs_diff * abs_diff / beta
        else:
            loss[i] = abs_diff - 0.5 * beta
    return {
        "none": loss.reshape(original_shape),
        "mean": np.mean(loss),
        "sum": np.sum(loss),
    }


def _np_smoothl1loss_grad(np_input, np_target, beta=1.0):
    original_shape = np_input.shape
    elem_cnt = np_input.size
    np_input = np_input.reshape(-1)
    np_target = np_target.reshape(-1)
    np_input_grad = np.zeros(elem_cnt).astype(np_input.dtype)
    for i in np.arange(elem_cnt):
        diff = np_input[i] - np_target[i]
        abs_diff = abs(diff)
        if abs_diff < beta:
            np_input_grad[i] = diff / beta
        else:
            np_input_grad[i] = np.sign(diff)
    np_input_grad_sum = np_input_grad.reshape(original_shape)
    np_input_grad_mean = np_input_grad_sum / elem_cnt
    return {
        "none": np_input_grad_sum,
        "mean": np_input_grad_mean,
        "sum": np_input_grad_sum,
    }


def _test_smoothl1loss_impl(test_case, device, shape, data_type, reduction, beta):
    x = np.random.randn(*shape).astype(type_name_to_np_type[data_type])
    y = np.random.randn(*shape).astype(type_name_to_np_type[data_type])
    input = flow.Tensor(
        x,
        dtype=type_name_to_flow_type[data_type],
        requires_grad=True,
        device=flow.device(device),
    )
    target = flow.Tensor(
        y, dtype=type_name_to_flow_type[data_type], device=flow.device(device)
    )
    loss = flow.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    loss = loss.to(device)
    of_out = loss(input, target)
    np_out = _np_smoothl1loss(x, y, beta)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_smoothl1loss_grad(x, y, beta=beta)[reduction]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestSmoothL1LossModule(flow.unittest.TestCase):
    def test_smoothl1loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_smoothl1loss_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(10, 3), (100,)]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["reduction"] = ["none", "mean", "sum"]
        arg_dict["beta"] = [0, 0.5, 1]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
