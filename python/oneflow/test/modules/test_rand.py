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
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgList


def _test_rand(test_case, device, shape):
    y1 = flow.rand(*shape, device=flow.device(device))
    y2 = flow.rand(size=shape, device=flow.device(device))

    test_case.assertTrue(not np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertTrue(shape == y1.shape)
    test_case.assertTrue(shape == y2.shape)


def _test_rand_tuple_shape(test_case, device, shape):
    y1 = flow.rand(shape, device=flow.device(device))
    y2 = flow.rand(shape, device=flow.device(device))

    test_case.assertTrue(not np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertTrue(shape == y1.shape)


def _test_0d_rand(test_case, device, shape):
    y1 = flow.rand(*shape, device=flow.device(device))
    y2 = flow.rand(*shape, device=flow.device(device))
    test_case.assertTrue(
        np.allclose(y1.numpy(), y2.numpy(), atol=1e-4, rtol=1e-4)
    )  # 0d is [] and []
    test_case.assertTrue(shape == y1.shape)


def _test_different_dtype(test_case, device, shape):
    y1 = flow.rand(*shape, dtype=flow.float32, device=flow.device(device))
    y2 = flow.rand(*shape, dtype=flow.float64, device=flow.device(device))
    test_case.assertTrue(not np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertTrue(shape == y1.shape)

    with test_case.assertRaises(NotImplementedError):
        flow.rand(*shape, dtype=flow.int32, device=flow.device(device))


def _test_backward(test_case, device, shape):
    x = flow.rand(*shape, device=flow.device(device), requires_grad=True)
    y = x.sum()
    y.backward()
    test_case.assertTrue(np.array_equal(np.ones(shape), x.grad.numpy()))


def _test_with_generator(test_case, device, shape):
    gen = flow.Generator()
    gen.manual_seed(0)
    y1 = flow.rand(
        *shape, dtype=flow.float32, device=flow.device(device), generator=gen
    )
    gen.manual_seed(0)
    y2 = flow.rand(
        *shape, dtype=flow.float32, device=flow.device(device), generator=gen
    )
    test_case.assertTrue(np.allclose(y1.numpy(), y2.numpy(), atol=1e-4, rtol=1e-4))


def _test_rand_with_flow_size(test_case, device, shape):
    y1 = flow.rand(flow.Size(shape), device=flow.device(device))
    y2 = flow.rand(flow.Size(shape), device=flow.device(device))

    test_case.assertTrue(not np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertTrue(shape == y1.shape)


@flow.unittest.skip_unless_1n1d()
class TestRandModule(flow.unittest.TestCase):
    def test_0d_randint(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_0d_rand]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 0, 4), (2, 0, 2)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_cases(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_rand,
            _test_rand_tuple_shape,
            _test_different_dtype,
            _test_backward,
            _test_with_generator,
            _test_rand_with_flow_size,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 4)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_half_rand(test_case):
        for device in ["cuda", "cpu"]:
            x = flow.rand(2, 3, dtype=flow.float16, device=flow.device(device))
            test_case.assertTrue(x.dtype == flow.float16)
            test_case.assertTrue(x.shape == flow.Size((2, 3)))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestRandOnNonDefaultDevice(flow.unittest.TestCase):
    def test_non_default_device(test_case):
        x = flow.rand(2, 3, device="cuda:1")
        test_case.assertEqual(x.device, flow.device("cuda:1"))


if __name__ == "__main__":
    unittest.main()
