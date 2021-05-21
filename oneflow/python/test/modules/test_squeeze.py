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


def _test_squeeze(test_case, device):
    input = flow.Tensor(
        np.array([[[[1, 1, 1]]]]).astype(np.int32), device=flow.device(device)
    )
    of_shape = flow.squeeze(input, dim=[1, 2]).numpy().shape
    np_shape = (1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(np.allclose(flow.squeeze(input, dim=[1, 2]).numpy(), np.squeeze(input.numpy(), axis=(1, 2)), 1e-4, 1e-4))


def _test_tensor_squeeze(test_case, device):
    input = flow.Tensor(
        np.array([[[[1, 1, 1]]]]).astype(np.int32), device=flow.device(device)
    )
    of_shape = input.squeeze(dim=[1, 2]).numpy().shape
    np_shape = (1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(np.allclose(input.squeeze(dim=[1, 2]).numpy(), np.squeeze(input.numpy(), axis=(1, 2)), 1e-4, 1e-4))


def _test_squeeze_int(test_case, device):
    input = flow.Tensor(
        np.array([[[[1, 1, 1]]]]).astype(np.int32), device=flow.device(device)
    )
    of_shape = flow.squeeze(input, 1).numpy().shape
    np_shape = (1, 1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(np.allclose(input.squeeze(1).numpy(), np.squeeze(input.numpy(), axis=1), 1e-4, 1e-4))


def _test_squeeze_backward(test_case, device):
    input = flow.Tensor(
        np.array([[[[1, 1, 1]]]]).astype(np.int32),
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.squeeze(input, dim=1).sum()
    y.backward()
    np_grad = [[[[1.0, 1.0, 1.0]]]]
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSqueeze(flow.unittest.TestCase):
    def test_squeeze(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_squeeze,
            _test_squeeze_int,
            _test_tensor_squeeze,
            _test_squeeze_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
