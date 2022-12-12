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


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensorOffload(flow.unittest.TestCase):
    def test_tensor_offload_d2h(test_case):
        input = flow.tensor(
            np.random.randn(1024, 1024, 100),
            dtype=flow.float32,
            device=flow.device("cuda"),
        )
        data = input.numpy()
        test_case.assertTrue(not input.is_offloaded())
        flow.cuda.empty_cache()
        before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print("cuda", before_used)

        # test tensor offload
        input.offload()
        test_case.assertTrue(input.is_offloaded())
        test_case.assertEqual(input.device, flow.device("cuda"))
        flow.cuda.empty_cache()
        after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print("cuda to cpu", after_used)
        # Check 400M cuda memory released
        test_case.assertTrue((before_used - after_used) == 400)

        # data = input.numpy() will raise error here

        # test tensor load
        input.load()
        test_case.assertTrue(not input.is_offloaded())
        test_case.assertEqual(input.device, flow.device("cuda"))
        flow.cuda.empty_cache()
        cur_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print("cpu to cuda", cur_used)
        # Check 400M cuda memory allocated
        test_case.assertTrue((cur_used - after_used) == 400)

        # test data after tensor load
        test_case.assertTrue(np.allclose(input.numpy(), data, rtol=0.0001, atol=0.0001))


if __name__ == "__main__":
    unittest.main()
