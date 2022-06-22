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


class TestModel1(flow.nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = flow.nn.Parameter(flow.ones(1))
        self.w2 = flow.nn.Parameter(flow.ones(1))

    def forward(self, x):
        y = x + self.w1
        y += 1
        y = y + self.w2
        return y


def display_all_pieces():
    flow.comm.barrier()
    flow._oneflow_internal.dtr.display_all_pieces()


def assert_memory(expected):
    flow.comm.barrier()
    assert flow._oneflow_internal.dtr.allocated_memory() == expected


def evict(tensor):
    flow.comm.barrier()
    flow._oneflow_internal.dtr.evict(tensor)


# TODO: add a pure cpu test
class TestDTR(flow.unittest.TestCase):
    def setUp(self):
        super().setUp()

        assert (
            os.getenv("ONEFLOW_DISABLE_VIEW") is not None
        ), "Please set ONEFLOW_DISABLE_VIEW to True, 1 or ON"
        assert (
            os.getenv("OF_DTR") is not None
        ), "Please set OF_DTR to True, 1 or ON"
        # wait for all previous operations to finish and
        # check the memory is empty at the beginning of every test case
        flow.comm.barrier()
        self.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 0)
        print(self._testMethodName)

    def test_dtr_enabled(test_case):
        flow.enable_dtr(True, "20KB", 0, "eq")
        test_case.assertTrue(flow.is_dtr_enabled())

    def test_dtr_work_on_simple_case_1(test_case):
        flow.enable_dtr(True, "20KB", 0, "eq")

        x1 = flow.ones(1024).to(
            "cuda"
        )  # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
        x2 = x1 + 3  # x2 = 4, total memory: 8KB
        x3 = x1 * x2  # x3 = 4, total memory: 12KB
        x4 = x1 - x3  # x4 = -3, total memory: 16KB
        x5 = x4.square()  # x5 = 9, total memory: 20KB
        x6 = x1 + x3  # x6 = 5, evict a tensor
        x7 = x1 + x3  # x7 = 5, evict a tensor

        # wait for the operations to finish
        flow.comm.barrier()
        # check if there are 2 tensors are evicted
        not_in_memory_num = sum(
            0 if flow._oneflow_internal.dtr.is_in_memory(x) else 1
            for x in [x1, x2, x3, x4, x5, x6, x7]
        )
        test_case.assertEqual(not_in_memory_num, 2)
        # check if the memory is full
        test_case.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 20 * 1024)

        # trigger recomputation and check the result
        y = x1 + x2 + x3 + x4 + x5 + x6 + x7

        test_case.assertTrue(np.array_equal(y.numpy(), 25 * np.ones(y.shape)))

    def test_dtr_work_on_simple_case_2(test_case):
        flow.enable_dtr(True, "16KB", 0, "eq")

        x1 = flow.ones(1024).to(
            "cuda"
        )  # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
        x2 = x1 + 3  # x2 = 4, total memory: 8KB
        x3 = x2 - 5  # x3 = -1, total memory: 12KB
        x4 = x3.relu()  # x4 = 0, evict a tensor
        x5 = x2.square()  # x5 = 16, evict a tensor

        # wait for the operations to finish
        flow.comm.barrier()
        # check if there is 1 tensors are evicted
        not_in_memory_num = sum(
            0 if flow._oneflow_internal.dtr.is_in_memory(x) else 1
            for x in [x1, x2, x3, x4, x5]
        )
        test_case.assertEqual(not_in_memory_num, 1)
        # check if the memory is full
        test_case.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 16 * 1024)

        # trigger recomputation and check the result
        y = x1 + x2 + x3 + x4 + x5

        test_case.assertTrue(np.array_equal(y.numpy(), 20 * np.ones(y.shape)))

    def test_evict_api(test_case):
        flow.enable_dtr(True, "36KB", 0, "eq")
        x1 = flow.ones(1024).to("cuda")
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

        m = flow.nn.BatchNorm2d(1024).to("cuda")

        x1 = flow.reshape(
            flow.rand(1024).to("cuda"), (1, 1024, 1, 1)
        )  # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
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

    def test_dtr_work_on_inplace(test_case):
        flow.enable_dtr(True, "12KB", 0, "eq")

        x1 = flow.ones(1024, requires_grad=True).to("cuda")  # 4KB (x1=1)
        x2 = x1 * 2  # 8KB (x1=1, x2=2)
        y = x2 + 1  # 12KB (x1=1, x2=2, y=3)
        x2.add_(1)  # 12KB (x1=1, x2=3, y=3)
        test_case.assertEqual(x2.grad_fn.name(), "scalar_add_backward")
        flow.comm.barrier()

        x3 = x2 + 1
        x4 = x2 + 1
        flow.comm.barrier()
        test_case.assertFalse(
            flow._oneflow_internal.dtr.is_in_memory(y)
        )  # make sure y is evicted
        x5 = y + 1

        # If inplace right in DTR, y should be recomputed as 3 and x5 should be 4. Otherwise, y could be 4 and x5 could be 5
        test_case.assertTrue(np.array_equal(x5.numpy(), 4 * np.ones(x5.shape)))

    def test_inplace_grad_fn(test_case):
        flow.enable_dtr(True, "2500MB", 0, "eq")

        m = TestModel1().to("cuda")
        x = flow.ones(1).requires_grad_().to("cuda")
        loss = m(x)
        loss.backward()

        test_case.assertEqual(type(m.w1.grad), type(x))
        test_case.assertEqual(type(m.w2.grad), type(x))

    @unittest.skipIf(
        not flow.support.env_var_util.parse_boolean_form_env("ONEFLOW_DTR_FBIP", True),
        reason="this test is for fbip",
    )
    def test_fbip1(test_case):
        flow.enable_dtr(True, "1MB", 0, "eq")

        a_np = np.concatenate((np.ones(512) * -3, np.ones(512))).astype(np.float32)
        c_np = np.concatenate((np.ones(512) * -1, np.ones(512) * 3)).astype(np.float32)
        e_np = np.concatenate((np.ones(512), np.ones(512) * 13)).astype(np.float32)
        a_grad_np = np.concatenate((np.zeros(512), np.ones(512) * 3)).astype(np.float32)

        a = flow.tensor(a_np, device="cuda").requires_grad_()
        assert a.shape == (1024,)
        assert_memory(1024 * 4)
        b = a + 1
        assert_memory(2048 * 4)
        c = b + 1
        assert_memory(3072 * 4)
        b_old_dptr = b.data_ptr()
        b *= 3
        assert_memory(3072 * 4)
        assert b.data_ptr() == b_old_dptr
        e = b + 7
        assert_memory(4096 * 4)
        d = flow.nn.functional.relu(b, inplace=True)
        assert_memory(4096 * 4)

        assert id(d) == id(b)
        assert d.data_ptr() == b.data_ptr()
        evict(c)
        assert_memory(3072 * 4)
        evict(e)
        assert_memory(2048 * 4)

        assert np.array_equal(c.numpy(), c_np), c.numpy()
        assert np.array_equal(e.numpy(), e_np), c.numpy()
        assert d.grad_fn.name() == "relu_backward"
        d.sum().backward()
        assert np.array_equal(a.grad.numpy(), a_grad_np), a.grad.numpy()

    @unittest.skip(
        reason="wait for zero_ kernel",
    )
    def test_fbip2(test_case):
        flow.enable_dtr(True, "1MB", 0, "eq")
        x1 = flow.ones(1024, device='cuda')
        print(flow._oneflow_internal.dtr.tensor_info(x1))
        x2 = x1 * 3
        x1.zero_()
        print(flow._oneflow_internal.dtr.tensor_info(x1))
        assert_memory(2048 * 4)
        x3 = x1 + 9
        assert_memory(3072 * 4)
        evict(x2)
        assert_memory(2048 * 4)
        evict(x1)
        assert_memory(1024 * 4)
        assert np.array_equal(x3.numpy(), np.ones(x3.shape) * 9)
        assert np.array_equal(x2.numpy(), np.ones(x3.shape) * 3)


if __name__ == "__main__":
    unittest.main()
