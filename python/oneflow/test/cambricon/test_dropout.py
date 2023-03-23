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
import os

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def do_test_dropout_numpy_p0(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(np.float32)
    np_one_mask = np.ones_like(np_x)
    x_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    out = flow._C.dropout(x_tensor, p=0.0)
    test_case.assertTrue(np.allclose(out.numpy(), np_x, atol=1e-5, rtol=1e-5))
    out_sum = out.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_numpy_p1(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    np_zero_mask = np.zeros_like(np_x)
    x_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    out = flow._C.dropout(x_tensor, p=1.0)
    test_case.assertTrue(np.allclose(out.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5))
    out_sum = out.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5)
    )


@flow.unittest.skip_unless_1n1d()
class TestMluDropout(flow.unittest.TestCase):
    def test_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            do_test_dropout_numpy_p0,
            do_test_dropout_numpy_p1,
        ]
        arg_dict["shape"] = [[2, 44, 66], [1, 2, 7], [5, 32, 74], [8, 125, 63]]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [np.float32]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
