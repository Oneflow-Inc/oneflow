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
import oneflow as flow
import numpy as np
import os
import random
import oneflow.typing as oft
from collections import OrderedDict
from test_util import (
    GenArgList,
    GenArgDict,
    type_name_to_flow_type,
    type_name_to_np_type,
)


def _compare_with_np(
    test_case, x_shape, dtype,
):
    x = np.random.randn(*x_shape).astype(type_name_to_np_type[dtype])
    ret = flow.Size(x_shape)
    for idx in range(0, len(ret)):
        test_case.assertEqual(ret[idx], x.shape[idx])


@flow.unittest.skip_unless_1n1d()
class TestSize(flow.unittest.TestCase):
    def test_size(test_case):
        size = flow.Size((4, 3, 10, 5))
        test_case.assertTrue(size[0] == 4)
        test_case.assertTrue(size[2] == 10)
        test_case.assertTrue(len(size) == 4)

        size = flow.Size([4, 3, 10, 5])
        test_case.assertTrue(size[0] == 4)
        test_case.assertTrue(size[2] == 10)
        test_case.assertTrue(len(size) == 4)

    def test_unpack(test_case):
        one, two, three, four = flow.Size((1, 2, 3, 4))
        test_case.assertEqual(one, 1)
        test_case.assertEqual(two, 2)
        test_case.assertEqual(three, 3)
        test_case.assertEqual(four, 4)

    def test_offical(test_case):
        arg_dict = OrderedDict()
        arg_dict["x_shape"] = [
            (10,),
            (20, 10),
            (20, 10, 10),
            (20, 10, 10, 3),
            (20, 10, 10, 3, 3),
        ]
        arg_dict["dtype"] = ["float32", "int32", "double"]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    def test_numel(test_case):
        size = flow.Size((1, 2, 3, 4))
        test_case.assertEqual(size.numel(), 24)

    def test_count(test_case):
        size = flow.Size((2, 2, 3, 4))
        test_case.assertEqual(size.count(1), 0)
        test_case.assertEqual(size.count(2), 2)
        test_case.assertEqual(size.count(3), 1)
        test_case.assertEqual(size.count(4), 1)

    def test_index(test_case):
        size = flow.Size((2, 3, 2, 4, 4))
        test_case.assertEqual(size.index(2), 0)
        test_case.assertEqual(size.index(2, start=0), 0)
        test_case.assertEqual(size.index(2, start=0, end=20), 0)
        test_case.assertEqual(size.index(2, start=1, end=20), 2)
        test_case.assertEqual(size.index(4), 3)
        test_case.assertEqual(size.index(4, start=4), 4)
        with test_case.assertRaises(ValueError):
            size.index(4, start=0, end=3)
        with test_case.assertRaises(ValueError):
            size.index(5)
        with test_case.assertRaises(ValueError):
            size.index(2, start=3)


if __name__ == "__main__":
    unittest.main()
