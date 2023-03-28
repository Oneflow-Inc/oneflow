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

import oneflow as flow

from oneflow.test_utils.test_util import GenArgDict


def _test_local_empty(test_case, shape, dtype, device, requires_grad):
    x = flow.empty(
        shape,
        dtype=dtype,
        device=flow.device(device),
        requires_grad=requires_grad if dtype == flow.float32 else False,
    )
    test_case.assertFalse(x.is_global)
    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.dtype, dtype)
    test_case.assertEqual(x.device, flow.device(device))
    if dtype == flow.float32:
        test_case.assertEqual(x.requires_grad, requires_grad)
    empty_like_x = flow.empty_like(
        x,
        dtype=dtype,
        device=flow.device(device),
        requires_grad=requires_grad if dtype == flow.float32 else False,
    )
    test_case.assertFalse(empty_like_x.is_global)
    test_case.assertEqual(empty_like_x.shape, flow.Size(shape))
    test_case.assertEqual(empty_like_x.dtype, dtype)
    test_case.assertEqual(empty_like_x.device, flow.device(device))
    if dtype == flow.float32:
        test_case.assertEqual(empty_like_x.requires_grad, requires_grad)


def _test_new_empty(test_case, shape, dtype, device, requires_grad):
    x = flow.empty(shape, dtype=dtype, device=flow.device(device))
    y = x.new_empty(
        shape,
        dtype=dtype,
        device=flow.device(device),
        requires_grad=requires_grad if dtype == flow.float32 else False,
    )
    test_case.assertFalse(y.is_global)
    test_case.assertEqual(y.shape, flow.Size(shape))
    test_case.assertEqual(y.dtype, dtype)
    test_case.assertEqual(y.device, flow.device(device))
    if dtype == flow.float32:
        test_case.assertEqual(y.requires_grad, requires_grad)

    y = x.new_empty(*shape)
    test_case.assertFalse(y.is_global)
    test_case.assertEqual(y.shape, flow.Size(shape))
    test_case.assertEqual(y.dtype, x.dtype)
    test_case.assertEqual(y.device, x.device)
    test_case.assertFalse(y.requires_grad)


def _test_local_empty_strided(test_case, shape, stride, dtype, device, requires_grad):
    x = flow.empty_strided(
        shape,
        stride,
        dtype=dtype,
        device=flow.device(device),
        requires_grad=requires_grad,
    )
    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.stride(), stride)
    test_case.assertEqual(x.dtype, dtype)
    test_case.assertEqual(x.device, flow.device(device))
    if dtype == flow.float32:
        test_case.assertEqual(x.requires_grad, requires_grad)


@flow.unittest.skip_unless_1n1d()
class TestEmptyOp(flow.unittest.TestCase):
    def test_local_empty(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["dtype"] = [flow.float32, flow.float16, flow.int32]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["requires_grad"] = [True, False]
        for arg in GenArgDict(arg_dict):
            _test_local_empty(test_case, **arg)
            _test_new_empty(test_case, **arg)

    def test_local_empty_strided(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 6), (2, 3, 12, 4)]
        arg_dict["stride"] = [(1, 2), (1, 2, 3), (2, 4, 5, 1)]
        arg_dict["dtype"] = [flow.float32, flow.float16, flow.int32]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["requires_grad"] = [True, False]
        for arg in GenArgDict(arg_dict):
            _test_local_empty_strided(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
