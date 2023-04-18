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


def _test_where(test_case, device):
    x = flow.tensor(
        np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
        dtype=flow.float32,
        device=flow.device(device),
    )
    y = flow.tensor(
        np.ones(shape=(3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.bool, device=flow.device(device)
    )
    of_out = flow.where(condition, x, y)
    np_out = np.array([[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_where_broadcast(test_case, device):
    x = flow.tensor(
        np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
        dtype=flow.float32,
        device=flow.device(device),
    )
    y = flow.tensor(
        np.ones(shape=(3, 3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[[0, 1], [1, 0], [1, 0]]]),
        dtype=flow.bool,
        device=flow.device(device),
    )
    of_out = flow.where(condition, x, y)
    np_out = np.array(
        [
            [[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]],
            [[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]],
            [[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]],
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_where_scalar(test_case, device):
    x = 0.5
    y = 2.0
    condition = flow.tensor(np.array([1]), dtype=flow.bool)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.float32)
    np_out = np.array([0.5])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    flow.set_default_dtype(flow.double)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.double)
    flow.set_default_dtype(flow.float16)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.float16)
    flow.set_default_dtype(flow.bfloat16)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.bfloat16)


def _test_where_dim4(test_case, device):
    x = flow.tensor(
        np.array([[[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]]),
        dtype=flow.float32,
        device=flow.device(device),
    )
    y = flow.tensor(
        np.ones(shape=(1, 1, 3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[[[0, 1], [1, 0], [1, 0]]]]),
        dtype=flow.bool,
        device=flow.device(device),
    )
    of_out = flow.where(condition, x, y)
    np_out = np.array([[[[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_where_scalar(test_case, device):
    x = flow.randn(400, 2000, device=device)
    y = flow.where(x > 0, x, 0.0)
    y_ref = np.where(x.numpy() > 0, x.numpy(), 0.0)
    test_case.assertTrue(np.allclose(y.numpy(), y_ref, 1e-05, 1e-05))

    y = flow.where(x > 0, 0.0, x)
    y_ref = np.where(x.numpy() > 0, 0.0, x.numpy())
    test_case.assertTrue(np.allclose(y.numpy(), y_ref, 1e-05, 1e-05))


def _test_where_scalar_tensor(test_case, device):
    scalar_zero = flow.Tensor(1, device=device).fill_(0.0)
    x = flow.randn(400, 2000, device=device)
    y = flow.where(x > 0, x, scalar_zero)
    y_ref = np.where(x.numpy() > 0, x.numpy(), 0.0)
    test_case.assertTrue(np.allclose(y.numpy(), y_ref, 1e-05, 1e-05))

    y = flow.where(x > 0, scalar_zero, x)
    y_ref = np.where(x.numpy() > 0, 0.0, x.numpy())
    test_case.assertTrue(np.allclose(y.numpy(), y_ref, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestWhereCambriconModule(flow.unittest.TestCase):
    def test_where(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_where,
            _test_where_broadcast,
            _test_where_scalar,
            _test_where_dim4,
            _test_where_scalar,
            _test_where_scalar_tensor,
        ]
        arg_dict["device"] = ["mlu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
