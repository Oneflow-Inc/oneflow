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
import sys
import re
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest


# @unittest.skipUnless(os.getenv("OF_DTR"), "only test while DTR is on")
class TestDTR(flow.unittest.TestCase):
    def setUp(self):
        super().setUp()
        
        assert os.getenv('ONEFLOW_DISABLE_VIEW') is not None, "Please set ONEFLOW_DISABLE_VIEW to True, 1 or ON"
        # wait for all previous operations to finish and 
        # check the memory is empty at the beginning of every test case
        flow.comm.barrier()
        self.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 0)

    def test_dtr_enabled(test_case):
        flow.enable_dtr(True, "20KB", 0, "eq")
        test_case.assertTrue(flow.is_dtr_enabled())

    def test_dtr_work_on_simple_case_1(test_case):
        flow.enable_dtr(True, "20KB", 0, "eq")

        x1 = flow.ones(1024).to('cuda') # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
        x2 = x1 + 3                     # x2 = 4, total memory: 8KB
        x3 = x1 * x2                    # x3 = 4, total memory: 12KB
        x4 = x1 - x3                    # x4 = -3, total memory: 16KB
        x5 = x4.square()                # x5 = 9, total memory: 20KB
        x6 = x1 + x3                    # x6 = 5, evict a tensor
        x7 = x1 + x3                    # x7 = 5, evict a tensor

        # wait for the operations to finish
        flow.comm.barrier()
        # check if there are 2 tensors are evicted
        not_in_memory_num = sum(0 if flow._oneflow_internal.dtr.is_in_memory(x) else 1 for x in [x1, x2, x3, x4, x5, x6, x7])
        test_case.assertEqual(not_in_memory_num, 2)
        # check if the memory is full
        test_case.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 20 * 1024)

        # trigger recomputation and check the result
        y = x1 + x2 + x3 + x4 + x5 + x6 + x7

        test_case.assertTrue(np.array_equal(y.numpy(), 25 * np.ones(y.shape)))

    def test_dtr_work_on_simple_case_2(test_case):
        flow.enable_dtr(True, "16KB", 0, "eq")

        x1 = flow.ones(1024).to('cuda') # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
        x2 = x1 + 3                     # x2 = 4, total memory: 8KB
        x3 = x2 - 5                     # x3 = -1, total memory: 12KB
        x4 = x3.relu()                  # x4 = 0, evict a tensor
        x5 = x2.square()                # x5 = 16, evict a tensor

        # wait for the operations to finish
        flow.comm.barrier()
        # check if there is 1 tensors are evicted
        not_in_memory_num = sum(0 if flow._oneflow_internal.dtr.is_in_memory(x) else 1 for x in [x1, x2, x3, x4, x5])
        test_case.assertEqual(not_in_memory_num, 1)
        # check if the memory is full
        test_case.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 16 * 1024)

        # trigger recomputation and check the result
        y = x1 + x2 + x3 + x4 + x5

        test_case.assertTrue(np.array_equal(y.numpy(), 20 * np.ones(y.shape)))

    def test_evict_api(test_case):
        flow.enable_dtr(True, "36KB", 0, "eq")
        x1 = flow.ones(1024).to('cuda')
        x2 = x1 + 1
        flow.comm.barrier()
        flow._oneflow_internal.dtr.evict(x2)
        test_case.assertFalse(flow._oneflow_internal.dtr.is_in_memory(x2))

        flow._oneflow_internal.dtr.evict(x2)
        test_case.assertFalse(flow._oneflow_internal.dtr.is_in_memory(x2))

        x3 = x2 + 1
        flow.comm.barrier()
        # NOTE: why is it false? because eager eviction in remat
        test_case.assertFalse(flow._oneflow_internal.dtr.is_in_memory(x2))
        test_case.assertTrue(flow._oneflow_internal.dtr.is_in_memory(x3))
        test_case.assertTrue(np.array_equal(x3, np.ones(x3.shape) * 3))
        test_case.assertTrue(np.array_equal(x2, np.ones(x2.shape) * 2))

    # def test_dropout(test_case):
    #     flow.enable_dtr(True, "120KB", 0, "eq")
    #     m = flow.nn.Dropout(p=0.5)
    #
    #     x1 = flow.rand(10).to('cuda')
    #     test_case.assertTrue(flow._oneflow_internal.dtr.is_dtr_tensor(x1))
    #     x2 = m(x1)
    #     test_case.assertTrue(flow._oneflow_internal.dtr.is_dtr_tensor(x2))
    #
    #     flow.comm.barrier()
    #     test_case.assertTrue(flow._oneflow_internal.dtr.is_in_memory(x2))
    #
    #     print('hahahaha')
    #
    #     x2_np1 = x2.numpy()
    #     print('2ahahaha')
    #     flow._oneflow_internal.dtr.evict(x2)
    #     print('0ahahaha')
    #     test_case.assertFalse(flow._oneflow_internal.dtr.is_in_memory(x2))
    #
    #     test_case.assertTrue(flow._oneflow_internal.dtr.is_in_memory(x1))
    #
    #     x2_np2 = x2.numpy()
    #     print('3ahahaha')
    #
    #     print(x2_np1)
    #     print(x2_np2)
    #     test_case.assertTrue(np.array_equal(x2_np1, x2_np2))

    def test_bn(test_case):
        flow.enable_dtr(True, "120KB", 0, "eq")

        m = flow.nn.BatchNorm2d(1024).to('cuda')

        x1 = flow.reshape(flow.rand(1024).to('cuda'), (1, 1024, 1, 1)) # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
        x2 = m(x1)
        test_case.assertTrue(flow._oneflow_internal.dtr.is_dtr_tensor(x2))
        test_case.assertTrue(flow._oneflow_internal.dtr.is_dtr_tensor(x1))
        rm_np1 = m.running_mean.numpy()
        test_case.assertTrue(flow._oneflow_internal.dtr.is_in_memory(x2))
        x2_np1 = x2.numpy()

        flow.comm.barrier()
        test_case.assertTrue(flow._oneflow_internal.dtr.is_in_memory(x2))

        flow._oneflow_internal.dtr.evict(x2)
        test_case.assertFalse(flow._oneflow_internal.dtr.is_in_memory(x2))
        x2_np2 = x2.numpy()
        rm_np2 = m.running_mean.numpy()

        flow._oneflow_internal.dtr.evict(x2)
        test_case.assertFalse(flow._oneflow_internal.dtr.is_in_memory(x2))
        x2_np3 = x2.numpy()
        rm_np3 = m.running_mean.numpy()
        test_case.assertTrue(np.array_equal(x2_np1, x2_np2))
        test_case.assertTrue(np.array_equal(x2_np1, x2_np3))
        test_case.assertTrue(np.array_equal(rm_np1, rm_np2))
        test_case.assertTrue(np.array_equal(rm_np1, rm_np3))

    # def test_dtr_threshold(test_case):
    #     regex = re.compile(r"(\d+(?:\.\d+)?)\s*([kmg]?b)", re.IGNORECASE)
    #     magnitude = ["b", "kb", "mb", "gb"]
    #     out = regex.findall(TestDTR.THRES)
    #     test_case.assertEqual(len(out), 1)
        

if __name__ == "__main__":
    unittest.main()
