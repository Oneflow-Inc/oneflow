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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_dropout(test_case, shape, device):
    input_arr = np.random.randn(*shape)
    m = flow.nn.Dropout(p=0)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), input_arr))


def _test_dropout_p1(test_case, shape, device):
    input_arr = np.random.randn(*shape)
    m = flow.nn.Dropout(p=1.0)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(
        np.allclose(y.numpy(), np.zeros(input_arr.shape, dtype=np.float32))
    )


def _test_dropout_backward_p0(test_case, shape, device):
    input_arr = np.random.randn(*shape)
    m = flow.nn.Dropout(p=0)
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(
            x.grad.numpy(), np.ones(input_arr.shape, dtype=np.float32), 1e-05, 1e-05
        )
    )


def _test_dropout_backward_p1(test_case, shape, device):
    input_arr = np.random.randn(*shape)
    m = flow.nn.Dropout(p=1)
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(
            x.grad.numpy(), np.zeros(input_arr.shape, dtype=np.float32), 1e-05, 1e-05
        )
    )


def _test_dropout_eval(test_case, shape, device):
    input_arr = np.random.randn(*shape)
    m = flow.nn.Dropout(p=1)
    x = flow.Tensor(input_arr, device=flow.device(device))
    m.eval()
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), input_arr))


def _test_dropout_with_generator(test_case, shape, device):
    generator = flow.Generator()
    generator.manual_seed(0)
    m = flow.nn.Dropout(p=0.5, generator=generator)
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y_1 = m(x)
    y_1.numpy()
    generator.manual_seed(0)
    y_2 = m(x)
    test_case.assertTrue(np.allclose(y_1.numpy(), y_2.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestDropout(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [
            _test_dropout,
            _test_dropout_p1,
            _test_dropout_backward_p0,
            _test_dropout_backward_p1,
            _test_dropout_eval,
            _test_dropout_with_generator,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
