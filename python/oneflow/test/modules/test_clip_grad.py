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
import os
import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList


def _clip_grad_norm_np(input, max_norm, norm_type):
    np_out = np.maximum(0, input)
    np_grad = np.array(np_out > 0, dtype=np.float32)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    input = [input]
    if len(input) == 0:
        return 0, 0
    if norm_type == float("inf"):
        total_norm = np.max(np.abs(np_grad))
    if norm_type == float("-inf"):
        total_norm = np.min(np.abs(np_grad))
    elif norm_type == 0:
        total_norm = np.sum(np.stack([np.sum(np_grad != 0)]) != 0)
    else:
        total_norm = np_grad
        for i in range(np_grad.ndim, 0, -1):
            total_norm = np.linalg.norm(total_norm, norm_type, axis=i - 1)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        np_grad = np.dot(np_grad, clip_coef)
    return total_norm, np_grad


def _test_clip_grad_norm_impl(test_case, shape, device, max_norm, norm_type):
    np_input = np.random.rand(*shape)
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = flow.nn.ReLU()
    of_out = m(of_input)
    of_out = of_out.sum()
    of_out.backward()
    of_total_norm = flow.nn.utils.clip_grad_norm_(of_input, max_norm, norm_type)
    np_total_norm, np_grad = _clip_grad_norm_np(np_input, max_norm, norm_type)
    test_case.assertTrue(
        np.allclose(of_total_norm.numpy(), np_total_norm, 1e-4, 1e-4, equal_nan=True)
    )
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_grad, 1e-4, 1e-4, equal_nan=True)
    )


def _clip_grad_value_np(input, clip_value):
    np_out = np.maximum(0, input)
    np_grad = np.array(np_out > 0, dtype=np.float32)
    clip_value = float(clip_value)
    if len(input) == 0:
        return 0, 0
    np_grad = np.clip(np_grad, -clip_value, clip_value)
    return np_grad


def _test_clip_grad_value_impl(test_case, shape, device, clip_value):
    np_input = np.random.rand(*shape)
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m = flow.nn.ReLU()
    of_out = m(of_input)
    of_out = of_out.sum()
    of_out.backward()
    flow.nn.utils.clip_grad_value_(of_input, clip_value)
    of_grad = of_input.grad.numpy()
    np_grad = _clip_grad_value_np(np_input, clip_value)
    test_case.assertTrue(np.allclose(of_grad, np_grad, 1e-4, 1e-4, equal_nan=True))


class ReluGraph(flow.nn.Graph):
    def __init__(self, clip_value) -> None:
        super().__init__()
        self.clip_value = clip_value

    def build(self, x):
        flow.nn.utils.clip_grad_value_(x, self.clip_value)
        return x


def _test_graph_clip_grad_value_impl(test_case, shape, device, clip_value):
    np_input = np.random.rand(*shape)
    of_input = flow.tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_eager_out = of_input
    flow.nn.utils.clip_grad_value_(of_eager_out, clip_value)
    relu_graph = ReluGraph(clip_value)
    of_graph_out = relu_graph(of_input)
    test_case.assertTrue(
        np.allclose(
            of_eager_out.numpy(), of_graph_out.numpy(), 1e-4, 1e-4, equal_nan=True
        )
    )


def _test_clip_grad_norm_global_impl(
    test_case, shape, sbp, placement, max_norm, norm_type
):
    of_input = flow.rand(
        *shape, dtype=flow.float32, sbp=sbp, placement=placement, requires_grad=True
    )
    np_input = of_input.to_global(sbp=flow.sbp.broadcast).to_local().numpy()

    m = flow.nn.ReLU()
    of_out = m(of_input)
    of_out = of_out.sum()
    of_out.backward()
    of_total_norm = flow.nn.utils.clip_grad_norm_(
        of_input, max_norm, norm_type
    ).to_local()
    np_total_norm, np_grad = _clip_grad_norm_np(np_input, max_norm, norm_type)
    test_case.assertTrue(
        np.allclose(of_total_norm.numpy(), np_total_norm, 1e-4, 1e-4, equal_nan=True)
    )
    test_case.assertTrue(
        np.allclose(
            of_input.grad.to_global(sbp=flow.sbp.broadcast).to_local().numpy(),
            np_grad,
            1e-4,
            1e-4,
            equal_nan=True,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestClipGrad(flow.unittest.TestCase):
    def test_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["max_norm"] = [0, 0.5, 1.0]
        arg_dict["norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        for arg in GenArgList(arg_dict):
            _test_clip_grad_norm_impl(test_case, *arg)

    def test_clip_value(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["clip_value"] = [0, 0.5, 1.0]
        for arg in GenArgList(arg_dict):
            _test_clip_grad_value_impl(test_case, *arg)
            _test_graph_clip_grad_value_impl(test_case, *arg)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestClipGradGlobal(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_clip_grad_global(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 4), (2, 4, 3), (2, 4, 5, 6)]
        arg_dict["sbp"] = [flow.sbp.broadcast, flow.sbp.split(0), flow.sbp.split(1)]
        arg_dict["placement"] = [
            flow.placement.all("cpu"),
            flow.placement.all("cuda"),
        ]
        arg_dict["max_norm"] = [0, 0.5, 1.0]
        arg_dict["norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        for arg in GenArgList(arg_dict):
            _test_clip_grad_norm_global_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
