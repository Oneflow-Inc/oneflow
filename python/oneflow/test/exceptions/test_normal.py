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
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestNormalError(flow.unittest.TestCase):
    def test_normal_data_type_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow._C.normal(mean=0.0, std=1.0, size=(3, 3), dtype=flow.int32)

        test_case.assertTrue(
            "Only support float and double in normal()." in str(ctx.exception)
        )

    def test_normal_out_tensor_data_type_error(test_case):
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.zeros((3, 3), dtype=flow.float64)
            x = flow._C.normal(
                mean=0.0, std=1.0, size=(3, 3), dtype=flow.float32, out=out
            )

        test_case.assertTrue(
            "data type oneflow.float32 does not match data type of out parameter oneflow.float64"
            in str(ctx.exception)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_normal_out_tensor_device_type_error(test_case):
        with test_case.assertRaises(RuntimeError) as ctx:
            out = flow.zeros((3, 3), dtype=flow.float32, device="cuda")
            x = flow._C.normal(
                mean=0.0,
                std=1.0,
                size=(3, 3),
                dtype=flow.float32,
                out=out,
                device="cpu",
            )

        test_case.assertTrue(
            "device type cpu:0 does not match device type of out parameter cuda:0"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
