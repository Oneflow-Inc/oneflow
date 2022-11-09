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
import numpy as np
from collections import OrderedDict

from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type
from oneflow.test_utils.automated_test_util import *
import oneflow as flow


def _test_normal(test_case, mean, std, shape, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    y1 = flow.normal(mean, std, shape, dtype=dtype, device=flow.device(device))
    y2 = flow.normal(mean, std, size=shape, dtype=dtype, device=flow.device(device))
    test_case.assertFalse(np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertEqual(shape, y1.shape)
    test_case.assertEqual(dtype, y1.dtype)
    test_case.assertEqual(shape, y2.shape)
    test_case.assertEqual(dtype, y2.dtype)


def _test_with_generator(test_case, mean, std, shape, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    gen = flow.Generator()
    gen.manual_seed(0)
    y1 = flow.normal(
        mean, std, shape, generator=gen, dtype=dtype, device=flow.device(device)
    )
    gen.manual_seed(0)
    y2 = flow.normal(
        mean, std, shape, generator=gen, dtype=dtype, device=flow.device(device)
    )
    test_case.assertTrue(np.array_equal(y1.numpy(), y2.numpy()))


def _test_backward(test_case, mean, std, shape, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    x = flow.normal(
        mean, std, shape, dtype=dtype, device=flow.device(device), requires_grad=True
    )
    y = x.sum()
    y.backward()
    test_case.assertTrue(np.array_equal(np.ones(shape), x.grad.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestNormModule(flow.unittest.TestCase):
    def test_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [_test_normal, _test_with_generator, _test_backward]
        arg_dict["mean"] = [-1, 0, 1]
        arg_dict["std"] = [1, 2, 8]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
