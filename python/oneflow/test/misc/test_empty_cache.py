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
import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestEmptyCache(flow.unittest.TestCase):
    def test_cuda_to_cpu_empty_cache(test_case):
        if flow._oneflow_internal.flags.with_cuda():

            x = flow.randn(512, 3, 512, 512).to("cuda")
            used_mem1 = flow._oneflow_internal.GetCUDAMemoryUsed()

            x = x.cpu()
            used_mem2 = flow._oneflow_internal.GetCUDAMemoryUsed()

            flow.cuda.empty_cache()
            used_mem3 = flow._oneflow_internal.GetCUDAMemoryUsed()
            test_case.assertTrue((used_mem3 < used_mem1) and (used_mem3 < used_mem2))


if __name__ == "__main__":
    unittest.main()
