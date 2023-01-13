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
from contextlib import contextmanager
import os
import sys
import re
import unittest
import functools

import numpy as np

import oneflow as flow
from oneflow import nn
import flowvision
import oneflow.unittest


def evict(tensor):
    flow._oneflow_internal.dtr.evict(tensor)


def is_in_memory(tensor):
    return flow._oneflow_internal.dtr.is_in_memory(tensor)


placeholder_size = 0

def allocated_memory(device, include_test_placeholder=False):
    return flow._oneflow_internal.dtr.allocated_memory(device) - (0 if include_test_placeholder else placeholder_size)


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


def only_fbip():
    if os.getenv("ONEFLOW_DTR_COPY_ON_WRITE") is None:
        return lambda f: f
    else:
        return unittest.skip("")


def only_copy_on_write():
    if os.getenv("ONEFLOW_DTR_COPY_ON_WRITE") is not None:
        return lambda f: f
    else:
        return unittest.skip("")


@contextmanager
def generate_placeholder(size_mb, device):
    global placeholder_size
    placeholder_size = size_mb * 1024 * 1024
    x = flow.zeros(placeholder_size, dtype=flow.int8, device=device)
    flow._oneflow_internal.dtr.disable_eviction(x)
    try:
        yield
    finally:
        del x
        placeholder_size = 0


def memory_budget(budget_mb, device):
    def deco(f):
        @functools.wraps(f)
        def new_f(*args, **kwargs):
            total_budget = int(os.environ['ONEFLOW_DTR_BUDGET_MB'])
            assert total_budget >= budget_mb, "Not enough memory budget"
            with generate_placeholder(total_budget - budget_mb, device):
                return f(*args, **kwargs)
        return new_f
    return deco


class TestDTR(flow.unittest.TestCase):
    def setUp(self):
        super().setUp()

        print(self._testMethodName)
        assert (
            os.getenv("ONEFLOW_VM_MULTI_THREAD") is not None
        ), "Please set ONEFLOW_VM_MULTI_THREAD to False, 0 or OFF"
        # check the memory is empty at the beginning of every test case
        if allocated_memory('cpu') > 0:
            print('allocated_memory(cpu):', allocated_memory('cpu'))
            display('cpu')
        if allocated_memory('cuda') > 0:
            print('allocated_memory(cuda):', allocated_memory('cuda'))
            display('cuda')

        self.assertEqual(allocated_memory('cpu'), 0)
        self.assertEqual(allocated_memory('cuda'), 0)

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @only_fbip()
    @memory_budget(12, 'cpu')
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

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @only_fbip()
    @memory_budget(12, 'cpu')
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

    @flow.unittest.skip_unless_1n1d()
    @unittest.skip("mutation other than inplace is not supported yet")
    @assert_no_small_piece_optimization
    @only_fbip()
    @memory_budget(12, 'cpu')
    def test_dtr_work_on_fbip_3(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        x2 = x1 * -2 # 8MB
        x1.zero_()
        evict(x2)
        print(x2.numpy())
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape) * -2))

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @only_fbip()
    @memory_budget(12, 'cuda')
    def test_dtr_work_on_fbip_4(self):
        x1 = flow.ones(1024 * 1024).to('cuda') # 4MB
        x2 = x1 + 1
        x2 += x1
        x3 = x2.relu()
        x4 = x3 + 1
        evict(x3)
        evict(x2)
        evict(x1)
        evict(x3)
        self.assertTrue(np.array_equal(x4.numpy(), np.ones(x4.shape) * 4))

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @memory_budget(12, 'cpu')
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

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @memory_budget(12, 'cpu')
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

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @memory_budget(12, 'cpu')
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

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @memory_budget(12, 'cpu')
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

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @memory_budget(12, 'cpu')
    def test_dtr_init_constant_and_scalar(self):
        x1 = flow.ones(1024, 1024)
        x2 = x1 + 1
        flow.nn.init.constant_(x1, 5.)  # type: ignore[arg-type]

        evict(x1)
        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape) * 5))

        evict(x1)
        evict(x2)
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape) * 2))

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @memory_budget(100, 'cuda')
    def test_bn_and_backward(self):
        model = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                ).to('cuda')
        for x in model.parameters():
            x.grad = flow.zeros_like(x).to('cuda')
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        x = flow.ones(4, 3, 224, 224).to('cuda')
        cpu_mem = allocated_memory('cpu')
        cuda_mem = allocated_memory('cuda')
        for _ in range(10):
            cpu_mem2 = allocated_memory('cpu')
            cuda_mem2 = allocated_memory('cuda')
            self.assertEqual(cpu_mem, cpu_mem2)
            self.assertEqual(cuda_mem, cuda_mem2)
            loss = model(x).sum()
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()

    def _test_resnet18(self, ddp, expected_loss):
        flow.manual_seed(flow.env.get_rank())
        device = 'cpu'

        model = flowvision.models.resnet18().to(device)
        if ddp:
            model = flow.nn.parallel.DistributedDataParallel(model, use_bucket=False)
        criterion = nn.CrossEntropyLoss().to(device)

        for x in model.parameters():
            x.grad = flow.zeros_like(x).to(device)
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        x = flow.rand(10, 3, 224, 224).to(device)
        target = flow.randint(low=0, high=1000, size=(x.shape[0],)).to(device).to(flow.int32)
        # NOTE: there is a bug in current implementation about random ops:
        # x1 = flow.rand(5)
        # x2 = x1 + 1
        # del x1   <--- we cannot block the eviction of x1 here because it is controlled by the user
        # evict(x2)
        # recompute(x2) <-- recomputing x2 triggers the recomputation of x1 and causes inconsistentness
        flow._oneflow_internal.dtr.disable_eviction(x)
        flow._oneflow_internal.dtr.disable_eviction(target)
        ITER_NUM = 5
        for i in range(ITER_NUM):
            print('start allocated_memory(cpu):', allocated_memory('cpu'))
            output = model(x)
            loss = criterion(output, target)
            del output
            print(loss.numpy().item())
            if i == 4:
                self.assertTrue(loss.numpy().item() in expected_loss)
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()
            print('end allocated_memory(cpu):', allocated_memory('cpu'))

        # check there is more than 10 recomputations each iteration
        # so the correctness check makes sense.
        self.assertGreater(flow._oneflow_internal.dtr.recomputation_num(), ITER_NUM * 10)

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @only_fbip()
    @memory_budget(220, 'cpu')
    def test_resnet18(self):
        self._test_resnet18(False, [0.6304041147232056])

    @flow.unittest.skip_unless_1n2d()
    @assert_no_small_piece_optimization
    @only_fbip()
    @memory_budget(220, 'cpu')
    def test_resnet18_ddp_1n2d(self):
        # 2 devices, 2 losses
        self._test_resnet18(True, [1.8890058994293213, 1.8992782831192017])

    @flow.unittest.skip_unless_1n1d()
    @assert_no_small_piece_optimization
    @only_copy_on_write()
    @memory_budget(20, 'cpu')
    def test_copy_on_write(self):
        x1 = flow.ones(1024 * 1024) # 4MB
        x2 = flow.ones(1024 * 1024)
        x2 += x1
        print(x2.numpy())


if __name__ == "__main__":
    unittest.main()
