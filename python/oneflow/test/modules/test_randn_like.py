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
import os
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.test_util import GenArgList


def _test_randn_like(test_case, device, shape):
    x = flow.randn(shape)
    y = flow.randn_like(x, device=flow.device(device))
    test_case.assertTrue(x.shape == y.shape)


def _test_0d_randn_like(test_case, device, shape):
    x = flow.randn(shape)
    y = flow.randn_like(x, device=flow.device(device))
    test_case.assertTrue(x.shape == y.shape)


def _test_different_dtype(test_case, device, shape):
    for dtype in [
        flow.float16,
        flow.float32,
        flow.float64,
        flow.double,
    ]:
        x = flow.randn(shape, dtype=dtype)
        y = flow.randn_like(x, dtype=dtype, device=flow.device(device))
        test_case.assertTrue(x.shape == y.shape)


def _test_with_generator(test_case, device, shape):
    gen = flow.Generator()
    gen.manual_seed(0)
    x = flow.randn(shape)
    y1 = flow.randn_like(
        x, dtype=flow.float32, device=flow.device(device), generator=gen
    )
    gen.manual_seed(0)
    x = flow.randn(shape)
    y2 = flow.randn_like(
        x, dtype=flow.float32, device=flow.device(device), generator=gen
    )
    test_case.assertTrue(np.allclose(y1.numpy(), y2.numpy(), atol=1e-4, rtol=1e-4))


def _test_0rank(test_case, device, shape):
    x = flow.randn(shape)
    y = flow.randn_like(x, device=flow.device(device))
    test_case.assertTrue(x.shape == y.shape)


@flow.unittest.skip_unless_1n1d()
class TestRandIntLike(flow.unittest.TestCase):
    def test_global_different_types(test_case):
        for dtype in [
            flow.float16,
            flow.float32,
            flow.float64,
            flow.double,
        ]:
            placement = flow.placement("cpu", ranks=[0])
            sbp = (flow.sbp.broadcast,)
            x_ = flow.randn((10, 1), dtype=dtype)
            x = flow.randn_like(x_, placement=placement, sbp=sbp, dtype=dtype)
            test_case.assertEqual(x.dtype, dtype)
            test_case.assertEqual(x.sbp, sbp)
            test_case.assertEqual(x.placement, placement)

    def test_randn_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_randn_like,
            _test_different_dtype,
            _test_with_generator,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_0d_randn_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_0d_randn_like]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 0, 4), (2, 0, 2)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_0rank_randn_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_0rank]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [()]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestRandIntLikeOnNonDefaultDevice(flow.unittest.TestCase):
    def test_non_default_device(test_case):
        x_ = flow.randn((2, 3))
        x = flow.randn_like(x_, device="cuda:1")
        test_case.assertEqual(x.device, flow.device("cuda:1"))


if __name__ == "__main__":
    unittest.main()
