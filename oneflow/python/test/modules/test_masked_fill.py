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
from test_util import GenArgList
from collections import OrderedDict
import numpy as np
import oneflow.experimental as flow


def _test_masked_fill(test_case, device):
    input_arr = np.array(
        [
            [
                [-0.13169311, 0.97277078, 1.23305363, 1.56752789],
                [-1.51954275, 1.87629473, -0.53301206, 0.53006478],
                [-1.38244183, -2.63448052, 1.30845795, -0.67144869],
            ],
            [
                [0.41502161, 0.14452418, 0.38968, -1.76905653],
                [0.34675095, -0.7050969, -0.7647731, -0.73233418],
                [-1.90089858, 0.01262963, 0.74693893, 0.57132389],
            ],
        ]
    )

    output = np.array(
        [
            [
                [-0.1316931, 8.7654321, 8.7654321, 8.7654321],
                [-1.5195428, 8.7654321, -0.5330121, 8.7654321],
                [-1.3824418, -2.6344805, 8.7654321, -0.6714487],
            ],
            [
                [8.7654321, 8.7654321, 8.7654321, -1.7690565],
                [8.7654321, -0.7050969, -0.7647731, -0.7323342],
                [-1.9008986, 8.7654321, 8.7654321, 8.7654321],
            ],
        ]
    )

    fill_value = 8.7654321  # random value e.g. -1e9 3.14

    input = flow.Tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    mask = flow.Tensor((input_arr > 0).astype(np.int8), dtype=flow.int).to(
        flow.device(device)
    )
    of_out = input.masked_fill(mask, value=fill_value)
    test_case.assertTrue(np.allclose(of_out.numpy(), output))

    input2 = flow.Tensor(input_arr, dtype=flow.float32, device=flow.device(device))
    mask2 = flow.Tensor((input_arr > 0).astype(np.int8), dtype=flow.int).to(
        flow.device(device)
    )
    of_out2 = flow.masked_fill(input2, mask, value=fill_value)
    test_case.assertTrue(np.allclose(of_out2.numpy(), output))


def _test_masked_fill_backward(test_case, device):
    input_arr = np.array(
        [
            [
                [-0.13169311, 0.97277078, 1.23305363, 1.56752789],
                [-1.51954275, 1.87629473, -0.53301206, 0.53006478],
                [-1.38244183, -2.63448052, 1.30845795, -0.67144869],
            ],
            [
                [0.41502161, 0.14452418, 0.38968, -1.76905653],
                [0.34675095, -0.7050969, -0.7647731, -0.73233418],
                [-1.90089858, 0.01262963, 0.74693893, 0.57132389],
            ],
        ]
    )

    output = np.array(
        [
            [
                [-0.1316931, 8.7654321, 8.7654321, 8.7654321],
                [-1.5195428, 8.7654321, -0.5330121, 8.7654321],
                [-1.3824418, -2.6344805, 8.7654321, -0.6714487],
            ],
            [
                [8.7654321, 8.7654321, 8.7654321, -1.7690565],
                [8.7654321, -0.7050969, -0.7647731, -0.7323342],
                [-1.9008986, 8.7654321, 8.7654321, 8.7654321],
            ],
        ]
    )

    fill_value = 8.7654321  # random value e.g. -1e9 3.14

    x = flow.Tensor(
        input_arr, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    mask = flow.Tensor((input_arr > 0).astype(np.int8), dtype=flow.int).to(
        flow.device(device)
    )
    y = x.masked_fill(mask, value=fill_value)
    y.retain_grad()
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.allclose(y.grad.numpy(), np.ones(input_arr.shape)))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMaskedFill(flow.unittest.TestCase):
    def test_masked_fill(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_masked_fill,
            _test_masked_fill_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
