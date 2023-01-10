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

import oneflow as flow
import oneflow.unittest


_source_op_list = [
    flow.ones,
    flow.zeros,
    flow.rand,
    flow.randn,
    flow.empty,
    flow.Tensor,
]


class TestDefaultDTypeInferface(oneflow.unittest.TestCase):
    def test_set_default_dtype(test_case):
        flow.set_default_dtype(flow.float32)
        test_case.assertEqual(flow.get_default_dtype(), flow.float32)

        flow.set_default_dtype(flow.float64)
        test_case.assertEqual(flow.get_default_dtype(), flow.float64)

        for op in _source_op_list:
            x = op((2, 3))
            test_case.assertEqual(x.dtype, flow.float64)
            x = op(2, 3)
            test_case.assertEqual(x.dtype, flow.float64)

        with test_case.assertRaises(Exception) as ctx:
            flow.set_default_dtype(flow.int32)
        test_case.assertTrue(
            "only floating-point types are supported as the default type"
            in str(ctx.exception)
        )

    def test_set_default_tensor_type(test_case):
        flow.set_default_dtype(flow.float32)
        test_case.assertEqual(flow.get_default_dtype(), flow.float32)

        # set default tensor type by TensorType
        flow.set_default_tensor_type(flow.DoubleTensor)
        test_case.assertEqual(flow.get_default_dtype(), flow.float64)
        for op in _source_op_list:
            x = op((2, 3))
            test_case.assertEqual(x.dtype, flow.float64)
            x = op(2, 3)
            test_case.assertEqual(x.dtype, flow.float64)

        # set default tensor type by TensorType string
        flow.set_default_tensor_type("oneflow.FloatTensor")
        test_case.assertEqual(flow.get_default_dtype(), flow.float32)
        for op in _source_op_list:
            x = op((2, 3))
            test_case.assertEqual(x.dtype, flow.float32)

    def test_behavior_for_oneflow_tensor(test_case):
        # float32 scope
        flow.set_default_dtype(flow.float32)
        test_case.assertEqual(flow.get_default_dtype(), flow.float32)

        x = flow.tensor([1.0, 2])
        test_case.assertEqual(x.dtype, flow.float32)

        # float64 scope
        flow.set_default_dtype(flow.float64)
        test_case.assertEqual(flow.get_default_dtype(), flow.float64)

        x = flow.tensor([1.0, 2])
        test_case.assertEqual(x.dtype, flow.float64)

        # no affect for int type
        x = flow.tensor((2, 3))
        test_case.assertEqual(x.dtype, flow.int64)

        # no affect for numpy array input
        nd_arr = np.array([1, 2, 3]).astype(np.float32)
        x = flow.tensor(nd_arr)
        test_case.assertEqual(x.dtype, flow.float32)


if __name__ == "__main__":
    unittest.main()
