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


def np_repeat(x, sizes):
    return np.tile(x, sizes)


def _test_repeat_new_dim(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 4, 1, 3), dtype=flow.float32, device=flow.device(device)
    )
    sizes = (4, 3, 2, 3, 3)
    np_out = np_repeat(input.numpy(), sizes)
    of_out = input.repeat(sizes=sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_repeat_same_dim(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    sizes = (4, 2, 3, 19)
    of_out = input.repeat(sizes=sizes)
    np_out = np_repeat(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_repeat_same_dim_int(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3), dtype=flow.int32, device=flow.device(device)
    )
    size_tensor = flow.Tensor(np.random.randn(4, 2, 3, 19))
    sizes = size_tensor.size()
    of_out = input.repeat(sizes=sizes)
    np_out = np_repeat(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out.astype(np.int32)))


def _test_repeat_new_dim_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 4, 1, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    sizes = (4, 3, 2, 3, 3)
    of_out = input.repeat(sizes=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [[216.0, 216.0, 216.0]],
            [[216.0, 216.0, 216.0]],
            [[216.0, 216.0, 216.0]],
            [[216.0, 216.0, 216.0]],
        ],
        [
            [[216.0, 216.0, 216.0]],
            [[216.0, 216.0, 216.0]],
            [[216.0, 216.0, 216.0]],
            [[216.0, 216.0, 216.0]],
        ],
    ]
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


def _test_repeat_same_dim_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    sizes = (1, 2, 3, 1)
    of_out = input.repeat(sizes=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [
        [
            [
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
            ],
            [
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0],
            ],
        ]
    ]
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestRepeat(flow.unittest.TestCase):
    def test_repeat(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_repeat_new_dim,
            _test_repeat_same_dim,
            _test_repeat_same_dim_int,
            _test_repeat_new_dim_backward,
            _test_repeat_same_dim_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
