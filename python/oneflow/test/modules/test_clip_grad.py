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

import oneflow as flow
from test_util import GenArgList


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
    of_input = flow.Tensor(
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


@flow.unittest.skip_unless_1n1d()
class TestAcosh(flow.unittest.TestCase):
    def test_acosh(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["max_norm"] = [0, 0.5, 1.0]
        arg_dict["norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        for arg in GenArgList(arg_dict):
            _test_clip_grad_norm_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
