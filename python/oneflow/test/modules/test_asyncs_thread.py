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
class TestLocalThread(flow.unittest.TestCase):
    def test_stream(test_case):
        with flow.asyncs.thread(flow.asyncs.Thread()):
            test_case.assertEqual(flow.ones(1)[0], 1)


@flow.unittest.skip_unless_1n2d()
class TestGlobalThread(flow.unittest.TestCase):
    def test_cpu_stream(test_case):
        threads = [flow.asyncs.Thread() for i in range(7)]
        iter_and_threads = [(i, threads[i % 7]) for i in range(30)]
        for i, thread in iter_and_threads:
            with flow.asyncs.thread(thread):
                placement = flow.placement("cpu", [0, 1])
                tensor = flow.ones(2, placement=placement, sbp=flow.sbp.split(0))
                test_case.assertEqual(tensor[0], 1)
                test_case.assertEqual(tensor[1], 1)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cuda_stream(test_case):
        threads = [flow.asyncs.Thread() for i in range(7)]
        iter_and_threads = [(i, threads[i % 7]) for i in range(200)]
        tensors = []
        dim = 0
        for i, thread in iter_and_threads:
            dim += 1
            with flow.asyncs.thread(thread):
                placement = flow.placement("cuda", [0, 1])
                ones = flow.ones(2 * dim, placement=placement, sbp=flow.sbp.split(0))
                tensors.append(ones.to_global(sbp=flow.sbp.broadcast) + i)
        for i, tensor in enumerate(tensors):
            test_case.assertEqual(tensor[0], 1 + i)
            test_case.assertEqual(tensor[int(tensor.shape[0] / 2)], 1 + i)


if __name__ == "__main__":
    unittest.main()
