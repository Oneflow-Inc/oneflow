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


@flow.unittest.skip_unless_1n1d()
class TestStream(flow.unittest.TestCase):
    def test_stream(test_case):
        with flow.stream(flow.Stream()):
            test_case.assertEqual(flow.ones(1)[0], 1)


@flow.unittest.skip_unless_1n2d()
class TestGlobalStream(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cpu_stream(test_case):
        worker_thread = flow.worker_thread(2)
        streams = [flow.Stream(worker_thread) for i in range(10)]
        for s in streams:
            with flow.stream(s):
                placement = flow.placement("cpu", [0, 1])
                tensor = flow.ones(2, placement=placement, sbp=flow.sbp.split(0))
                test_case.assertEqual(tensor[0], 1)
                test_case.assertEqual(tensor[1], 1)

    def test_cuda_stream(test_case):
        streams = [flow.Stream(flow.worker_thread(i % 4)) for i in range(200)]
        tensors = []
        dim = 0
        for s in streams:
            dim += 1
            with flow.stream(s):
                placement = flow.placement("cuda", [0, 1])
                ones = flow.ones(2 * dim, placement=placement, sbp=flow.sbp.split(0))
                tensors.append(ones.to_global(sbp=flow.sbp.broadcast))
        for tensor in tensors:
            test_case.assertEqual(tensor[0], 1)
            test_case.assertEqual(tensor[int(tensor.shape[0] / 2)], 1)


if __name__ == "__main__":
    unittest.main()
