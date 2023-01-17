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
import numpy as np
import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestNpDtypeConverter(flow.unittest.TestCase):
    def test_np_dtype_converter(test_case):
        for flow_dtype in flow.dtypes():
            if flow_dtype in [flow.record, flow.tensor_buffer, flow.bfloat16]:
                continue
            np_dtype = flow.convert_oneflow_dtype_to_numpy_dtype(flow_dtype)
            test_case.assertEqual(
                flow.framework.dtype.convert_numpy_dtype_to_oneflow_dtype(np_dtype),
                flow_dtype,
            )

            # Test whether dtype conversion works with arr.dtype
            np_arr = np.array([1, 2], dtype=np_dtype)
            test_case.assertEqual(np_arr.dtype, np_dtype)
            flow_tensor = flow.tensor([1, 2], dtype=flow_dtype)
            test_case.assertEqual(flow_tensor.dtype, flow_dtype)


if __name__ == "__main__":
    unittest.main()
