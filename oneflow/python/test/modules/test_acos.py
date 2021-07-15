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
from automated_test_util import *
import torch


def _test_acos_impl(test_case, shape, device):
    input = flow.Tensor(
        np.random.rand(*shape) - 0.5, device=flow.device(device), requires_grad=True
    )
    of_out = flow.acos(input)
    np_out = np.arccos(input.numpy())
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_grad = -1.0 / np.sqrt(1 - np.square(input.numpy()))
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-4, 1e-4, equal_nan=True)
    )


def acos_input_tensor(shape):
    def generator(_):
        low = -1
        high = 1
        rng = np.random.default_rng()
        np_arr = rng.random(size=shape) * (high - low) + low
        return (
            flow.Tensor(np_arr, dtype=flow.float32),
            torch.tensor(np_arr, dtype=torch.float32),
        )

    return generator


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAcos(flow.unittest.TestCase):
    def test_acos(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_acos_impl(test_case, *arg)

    def test_acos_flow_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            test_flow_against_pytorch(
                test_case,
                "acos",
                device=device,
                n=2,
                extra_generators={"input": acos_input_tensor((3, 3))},
            )

    def test_acos_tensor_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            test_tensor_against_pytorch(
                test_case,
                "acos",
                device=device,
                n=2,
                extra_generators={"input": acos_input_tensor((3, 3))},
            )


if __name__ == "__main__":
    unittest.main()
