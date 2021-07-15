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

import oneflow.experimental as flow
from test_util import GenArgList


def _prelu(input, alpha):
    alpha = np.expand_dims(alpha, 0)
    alpha = np.expand_dims(alpha, 2)
    alpha = np.expand_dims(alpha, 3)
    return np.where(input > 0, input, input * alpha)


def _prelu_grad(input, alpha):
    return alpha * (input <= 0) + (input > 0)


def _test_prelu(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    np_alpha = np.random.randn(1)
    prelu = flow.nn.PReLU(init=np_alpha)
    if device == "cuda":
        prelu.to(flow.device("cuda"))
    np_out = _prelu(np_input, np_alpha)
    of_out = prelu(input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_prelu_ndims(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    np_alpha = np.random.randn(shape[1])
    prelu = flow.nn.PReLU(init=1.0, num_parameters=shape[1])
    prelu_alpha = np.expand_dims(np_alpha, (1, 2))
    prelu.weight = flow.nn.Parameter(flow.Tensor(prelu_alpha, dtype=flow.float32))
    if device == "cuda":
        prelu.to(flow.device("cuda"))
    np_out = _prelu(np_input, np_alpha)
    of_out = prelu(input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_prelu_grad(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(
        np_input, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    np_alpha = 0.2
    prelu = flow.nn.PReLU(init=np_alpha)
    if device == "cuda":
        prelu.to(flow.device("cuda"))
    of_out = prelu(input).sum()
    of_out.backward()
    np_grad = _prelu_grad(np_input, np_alpha)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPReLU(flow.unittest.TestCase):
    def test_prelu(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_prelu(test_case, *arg)
            _test_prelu_ndims(test_case, *arg)
            _test_prelu_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
