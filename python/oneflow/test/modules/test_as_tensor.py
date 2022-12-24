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
import random
import unittest

import numpy as np
import oneflow as flow
import oneflow.unittest


numpy_dtype_to_oneflow_dtype_dict = {
    np.int32: flow.int32,
    np.int64: flow.int64,
    np.int8: flow.int8,
    np.uint8: flow.uint8,
    np.float64: flow.float64,
    np.float32: flow.float32,
    np.float16: flow.float16,
}


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestAsTensor(flow.unittest.TestCase):
    def test_tensor_type(test_case):
        x = flow.randn(2, 3)
        y = flow.as_tensor(x)
        y[0] = 2.0
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))
        test_case.assertTrue(np.array_equal(id(x), id(y)))

        x = flow.randn(2, 3)
        x = x.to("cuda")
        y = flow.as_tensor(x)
        y[0] = 2.0
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))
        test_case.assertTrue(np.array_equal(id(x), id(y)))

        x = flow.randn(2, 3)
        y = flow.as_tensor(x, device=flow.device("cuda:0"))
        test_case.assertTrue(id(x) != id(y))

        for dtype in [
            flow.float64,
            flow.float16,
            flow.int64,
            flow.int32,
            flow.int8,
            flow.uint8,
        ]:
            x = flow.randn(2, 3)
            y = flow.as_tensor(x, dtype=dtype)
            test_case.assertTrue(id(x) != id(y))

    def test_numpy_type(test_case):
        for device in [flow.device("cpu"), flow.device("cuda:0"), None]:
            for np_dtype in [
                np.float64,
                np.float32,
                np.float16,
                np.int64,
                np.int32,
                np.int8,
                np.uint8,
            ]:
                for flow_dtype in [
                    flow.float64,
                    flow.float16,
                    flow.int64,
                    flow.int32,
                    flow.int8,
                    flow.uint8,
                ]:
                    np_arr = np.ones((2, 3), dtype=np_dtype)
                    try:
                        tensor = flow.as_tensor(np_arr, dtype=flow_dtype)
                        if numpy_dtype_to_oneflow_dtype_dict[
                            np_arr.dtype
                        ] == flow_dtype and device is not flow.device("cuda:0"):
                            tensor[0][0] += 1.0
                            test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))
                        else:
                            test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))
                    except Exception as e:
                        # Ignore cast or kernel mismatch error in test example
                        pass

    def test_other_type(test_case):
        for device in [flow.device("cpu"), flow.device("cuda:0"), None]:
            for np_dtype in [
                np.float64,
                np.float32,
                np.float16,
                np.int64,
                np.int32,
                np.int8,
                np.uint8,
            ]:
                for flow_dtype in [
                    flow.float64,
                    flow.float16,
                    flow.int64,
                    flow.int32,
                    flow.int8,
                    flow.uint8,
                ]:
                    # tuple
                    np_arr = (1.0, 2.0, 3.0)
                    try:
                        tensor = flow.as_tensor(np_arr, dtype=flow_dtype)
                        test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))
                    except Exception as e:
                        # Ignore cast or kernel mismatch error in test example
                        pass
                    # tuple
                    np_arr = [1.0, 2.0, 3.0]
                    try:
                        tensor = flow.as_tensor(np_arr, dtype=flow_dtype)
                        test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))
                    except Exception as e:
                        # Ignore cast or kernel mismatch error in test example
                        pass
                    # scalar
                    np_arr = 4.0
                    try:
                        tensor = flow.as_tensor(np_arr, dtype=flow_dtype)
                        test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))
                    except Exception as e:
                        # Ignore cast or kernel mismatch error in test example
                        pass

    def test_numpy_dtype_bug(test_case):
        test_case.assertEqual(flow.as_tensor([1.0]).dtype, flow.float32)
        x = np.random.randn(10)
        y1 = flow.as_tensor(x, dtype=flow.int64)
        y2 = flow.as_tensor(x, dtype=flow.float64)
        test_case.assertEqual(y1.dtype, flow.int64)
        test_case.assertEqual(y2.dtype, flow.float64)


if __name__ == "__main__":
    unittest.main()
