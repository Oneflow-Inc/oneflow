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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_masked_select(test_case, device):
    x = flow.tensor(
        np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    mask = x.gt(0.05)
    of_out = flow.masked_select(x, mask)
    np_out = np.array([0.3139, 0.3898])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[0, 1], [1, 0], [0, 0]])
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_masked_select_broadcast(test_case, device):
    x = flow.tensor(
        np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    mask = flow.tensor(
        np.array(
            [
                [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[1.0, 0], [1.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        ),
        dtype=flow.int8,
        device=flow.device(device),
    )
    of_out = flow.masked_select(x, mask)
    np_out = [
        -0.462,
        0.3898,
        -0.7197,
        -0.1657,
        -0.462,
        0.3898,
        -0.7197,
        -0.1657,
        -0.462,
        0.3139,
        -0.7197,
        0.0478,
        -0.1657,
    ]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[3.0, 1.0], [2.0, 3.0], [1.0, 3.0]]]
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_masked_select_input_zero(test_case, device):
    x = flow.tensor(
        [[26, 14, 18, 14, 5, 18, 5, 18, 4, 18, 15, 18, 22, 18, 0]],
        device=flow.device(device),
        dtype=flow.int64,
    )
    f_mask = flow.tensor(
        [
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ]
        ],
        device=flow.device(device),
        dtype=flow.bool,
    )
    y = x.masked_select(f_mask)
    test_case.assertTrue(
        np.allclose(
            y.numpy(),
            [26, 14, 18, 14, 5, 18, 5, 18, 4, 18, 15, 18, 22, 18, 0],
            1e-05,
            1e-05,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestMaskedSelect(flow.unittest.TestCase):
    def test_masked_select(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_masked_select,
            _test_masked_select_broadcast,
            _test_masked_select_input_zero,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_masked_select_broadcast(test_case):
        x = flow.ones(2, 3, 3)
        mask = flow.triu(flow.ones(3, 3), 1)
        flow_res = flow.masked_select(x, mask)
        np_res = [1, 1, 1, 1, 1, 1]
        test_case.assertTrue(np.allclose(flow_res.numpy(), np_res, 1e-05, 1e-05))


if __name__ == "__main__":
    unittest.main()
