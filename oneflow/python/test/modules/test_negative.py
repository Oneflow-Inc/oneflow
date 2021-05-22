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
from collections import OrderedDict

import unittest
import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _test_negtive(test_case, device):
    input = flow.Tensor(
        np.array([1.0, -1.0, 2.3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.negative(input)
    np_out = -(input.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_negative_neg(test_case, device):
    input = flow.Tensor(
        np.array([1.0, -1.0, 2.3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.neg(input)
    np_out = -(input.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tensor_negative(test_case, device):
    input = flow.Tensor(
        np.array([1.0, -1.0, 2.3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = input.negative()
    np_out = -(input.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_self_tensor_negative(test_case, device):
    input = flow.Tensor(
        np.array([1.0, -1.0, 2.3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = -input
    np_out = -(input.numpy())
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_negative_backward(test_case, device):
    input = flow.Tensor(
        np.array([1.0, -1.0, 2.3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.negative(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [-1.0, -1.0, -1.0]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestNegativeModule(flow.unittest.TestCase):
    def test_negative(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_negtive,
            _test_negative_neg,
            _test_tensor_negative,
            _test_self_tensor_negative,
            _test_negative_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
