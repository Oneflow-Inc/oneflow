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
import functools

import numpy as np

import oneflow as flow
import oneflow.unittest


def evict(tensor):
    flow._oneflow_internal.dtr.evict(tensor)


def is_in_memory(tensor):
    return flow._oneflow_internal.dtr.is_in_memory(tensor)


def allocated_memory(device):
    return flow._oneflow_internal.dtr.allocated_memory(device)


def display(device):
    return flow._oneflow_internal.dtr.display(device)


def assert_no_small_piece_optimization(f):
    @functools.wraps(f)
    def new_f(*args, **kwargs):
        assert (
            os.getenv("ONEFLOW_DTR_SMALL_PIECE") is not None
        ), "Please set ONEFLOW_DTR_SMALL_PIECE to False, 0 or OFF"
        return f(*args, **kwargs)

    return new_f


class TestDTR(flow.unittest.TestCase):
    def setUp(self):
        super().setUp()

        print(self._testMethodName)
        assert (
            os.getenv("ONEFLOW_VM_NO_SCHEDULER_THREAD") is not None
        ), "Please set ONEFLOW_VM_NO_SCHEDULER_THREAD to True, 1 or ON"
        # check the memory is empty at the beginning of every test case
        if allocated_memory('cpu') > 0:
            display('cpu')
        if allocated_memory('cuda') > 0:
            display('cuda')

        self.assertEqual(allocated_memory('cpu'), 0)
        self.assertEqual(allocated_memory('cuda'), 0)

    @assert_no_small_piece_optimization
    def test_dtr_work_on_fbip_1(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        x2 = x1 * -2 # 8MB
        x3 = x2 - 2 # 12MB
        x2.relu_() # 12MB
        self.assertTrue(is_in_memory(x1))
        self.assertTrue(is_in_memory(x2))
        self.assertTrue(is_in_memory(x3))
        evict(x3)
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * -4))
        evict(x2)
        self.assertTrue(np.array_equal(x2.numpy(), np.zeros(x2.shape)))

    @assert_no_small_piece_optimization
    def test_dtr_work_on_fbip_2(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        x2 = x1[0]
        x3 = x2 + 2
        evict(x3)
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * 3))
        evict(x2)
        evict(x3)
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * 3))
        evict(x2)
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape)))

    @unittest.skip("mutation other than inplace is not supported yet")
    @assert_no_small_piece_optimization
    def test_dtr_work_on_fbip_3(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        x2 = x1 * -2 # 8MB
        x1.zero_()
        evict(x2)
        print(x2.numpy())
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape) * -2))

    @assert_no_small_piece_optimization
    def test_dtr_work_on_simple_case_1(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        self.assertTrue(is_in_memory(x1))
        self.assertEqual(allocated_memory('cpu'), 4 * 1024 * 1024)
        x2 = x1 + 2
        # eager eviction
        del x1
        self.assertTrue(is_in_memory(x2))
        x3 = x2 + 2
        self.assertTrue(is_in_memory(x2))
        x4 = x3 + 2
        self.assertTrue(is_in_memory(x2))
        x5 = x4 + 2
        self.assertFalse(is_in_memory(x2))
        self.assertTrue(is_in_memory(x3))
        x6 = x5 + 2
        self.assertFalse(is_in_memory(x2))
        self.assertFalse(is_in_memory(x3))

        self.assertTrue(np.array_equal(x6.numpy(), np.ones(x6.shape) * 11))
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * 5))

    @assert_no_small_piece_optimization
    def test_dtr_work_on_simple_case_2(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        self.assertTrue(is_in_memory(x1))
        self.assertEqual(allocated_memory('cpu'), 4 * 1024 * 1024)
        x2 = x1 + 2
        # eager eviction
        del x1
        self.assertTrue(is_in_memory(x2))
        x3 = x2 + 2
        self.assertTrue(is_in_memory(x2))
        x4 = x3 + 2
        self.assertTrue(is_in_memory(x2))
        x5 = x4 + 2
        self.assertFalse(is_in_memory(x2))
        self.assertTrue(is_in_memory(x3))
        x6 = x5 + 2
        self.assertFalse(is_in_memory(x2))
        self.assertFalse(is_in_memory(x3))

        self.assertTrue(np.array_equal(x6.numpy(), np.ones(x6.shape) * 11))
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * 5))

    @assert_no_small_piece_optimization
    def test_dtr_full_and_init_constant(self):
        x1 = flow.eye(1024, 1024) # 4MB
        self.assertTrue(is_in_memory(x1))
        self.assertEqual(allocated_memory('cpu'), 4 * 1024 * 1024)

        x2 = flow.full(x1.shape, 3.)
        flow.nn.init.constant_(x1, x2)  # type: ignore[arg-type]
        del x2
        self.assertEqual(allocated_memory('cpu'), 4 * 1024 * 1024)

        evict(x1)

        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape) * 3))

    @assert_no_small_piece_optimization
    def test_dtr_lifecycle_of_view_tensor(self):
        x1 = flow.eye(2, 3)
        self.assertTrue(is_in_memory(x1))

        x2 = flow.ones(3)
        x3 = flow.expand(x2, (2, 3))
        x1[:] = x3
        del x3
        del x2

        evict(x1)

        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape)))

    @assert_no_small_piece_optimization
    def test_dtr_init_constant_and_scalar(self):
        x1 = flow.ones(1024, 1024)
        x2 = x1 + 1
        flow.nn.init.constant_(x1, 5.)  # type: ignore[arg-type]

        evict(x1)
        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape) * 5))

        evict(x1)
        evict(x2)
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape) * 2))


if __name__ == "__main__":
    unittest.main()
