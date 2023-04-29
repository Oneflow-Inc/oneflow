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
"""
This file (_test_remat.py) is intended to be run inside test_remat.py
with correct environment variables like ONEFLOW_VM_MULTI_THREAD=0
"""
from contextlib import contextmanager
import os
import unittest
import functools

import numpy as np

import oneflow as flow
from oneflow import nn
import flowvision
import oneflow.unittest


def evict(tensor):
    flow._oneflow_internal.remat.evict(tensor)


def is_in_memory(tensor):
    return flow._oneflow_internal.remat.is_in_memory(tensor)


placeholder_size = 0


def allocated_memory(device, include_test_placeholder=False):
    if device == "cuda" and not flow.sysconfig.with_cuda():
        return 0
    return flow._oneflow_internal.remat.allocated_memory(device) - (
        0 if include_test_placeholder else placeholder_size
    )


def display(device):
    return flow._oneflow_internal.remat.display(device)


def only_fbip():
    if os.getenv("ONEFLOW_REMAT_COPY_ON_WRITE") is None:
        return lambda f: f
    else:
        return unittest.skip("")


def only_copy_on_write():
    if os.getenv("ONEFLOW_REMAT_COPY_ON_WRITE") is not None:
        return lambda f: f
    else:
        return unittest.skip("")


def loss_test():
    if os.getenv("ONEFLOW_REMAT_RUN_LOSS_TEST") is not None:
        return lambda f: f
    else:
        return unittest.skip(
            "Environment variable 'ONEFLOW_REMAT_RUN_LOSS_TEST' need to be set to run this test."
        )


@contextmanager
def generate_placeholder(size_mb, device):
    global placeholder_size
    placeholder_size = size_mb * 1024 * 1024
    x = flow.zeros(int(placeholder_size), dtype=flow.int8, device=device)
    flow._oneflow_internal.remat.disable_eviction(x)
    try:
        yield
    finally:
        del x
        placeholder_size = 0


def memory_budget(budget_mb, device):
    if device == "cuda" and not oneflow.sysconfig.with_cuda():
        return unittest.skip("Skip CUDA tests on CPU build")

    def deco(f):
        @functools.wraps(f)
        def new_f(*args, **kwargs):
            total_budget = flow.remat.get_budget() / 1024 / 1024
            assert total_budget >= budget_mb, "Not enough memory budget"
            remat_device = device + "+remat"
            with generate_placeholder(total_budget - budget_mb, remat_device):
                return f(*args, remat_device, **kwargs)

        return new_f

    return deco


class TestRemat(flow.unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flow.remat.set_budget("500MB")
        flow.remat.set_small_pieces_optimization(False)

    def setUp(self):
        super().setUp()

        assert (
            os.getenv("ONEFLOW_VM_MULTI_THREAD") is not None
        ), "Please set ONEFLOW_VM_MULTI_THREAD to False, 0 or OFF"
        # check the memory is empty at the beginning of every test case
        if allocated_memory("cpu") > 0:
            print("allocated_memory(cpu):", allocated_memory("cpu"))
            display("cpu")
        if allocated_memory("cuda") > 0:
            print("allocated_memory(cuda):", allocated_memory("cuda"))
            display("cuda")

        self.assertEqual(allocated_memory("cpu"), 0)
        self.assertEqual(allocated_memory("cuda"), 0)
        flow._oneflow_internal.remat.clear_stats()

    def tearDown(self):
        super().tearDown()
        # check the memory is empty at the end of every test case
        self.assertEqual(allocated_memory("cpu"), 0)
        self.assertEqual(allocated_memory("cuda"), 0)

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(12, "cpu")
    def test_remat_work_on_fbip_1(self, device):
        x1 = flow.ones(1024 * 1024, device=device)  # 4MB
        x2 = x1 * -2  # 8MB
        x3 = x2 - 2  # 12MB
        x2.relu_()  # 12MB
        self.assertTrue(is_in_memory(x1))
        self.assertTrue(is_in_memory(x2))
        self.assertTrue(is_in_memory(x3))
        evict(x3)
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * -4))
        evict(x2)
        self.assertTrue(np.array_equal(x2.numpy(), np.zeros(x2.shape)))

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(12, "cpu")
    def test_remat_work_on_fbip_2(self, device):
        x1 = flow.ones(1024 * 1024, device=device)  # 4MB
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
    @only_fbip()
    @memory_budget(12, "cpu")
    def test_remat_work_on_fbip_3(self, device):
        x1 = flow.ones(1024 * 1024, device=device)  # 4MB
        x2 = x1 * -2  # 8MB
        x1.zero_()
        evict(x2)
        print(x2.numpy())
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape) * -2))

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(12, "cuda")
    def test_remat_work_on_fbip_4(self, device):
        x1 = flow.ones(1024 * 1024, device=device)  # 4MB
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
    @memory_budget(12, "cpu")
    def test_remat_work_on_simple_case_1(self, device):
        x1 = flow.ones(1024 * 1024, device=device)  # 4MB
        self.assertTrue(is_in_memory(x1))
        self.assertEqual(allocated_memory(device), 4 * 1024 * 1024)
        x2 = x1 + 2
        self.assertEqual(allocated_memory(device), 8 * 1024 * 1024)
        # eager eviction
        del x1
        self.assertEqual(allocated_memory(device), 4 * 1024 * 1024)
        self.assertTrue(is_in_memory(x2))
        x3 = x2 + 2
        self.assertTrue(is_in_memory(x2))
        x4 = x3 + 2
        self.assertTrue(is_in_memory(x2))
        x5 = x4 + 2
        self.assertFalse(is_in_memory(x2))
        self.assertTrue(is_in_memory(x3))
        self.assertTrue(is_in_memory(x4))
        x6 = x5 + 2
        self.assertFalse(is_in_memory(x2))
        # the eviction of x2 increases the cost of x3, so x4 is evicted
        self.assertTrue(is_in_memory(x3))
        self.assertFalse(is_in_memory(x4))

        self.assertTrue(np.array_equal(x6.numpy(), np.ones(x6.shape) * 11))
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * 5))

    @flow.unittest.skip_unless_1n1d()
    @memory_budget(12, "cpu")
    def test_remat_work_on_simple_case_2(self, device):
        x1 = flow.ones(1024 * 1024, device=device)  # 4MB
        self.assertTrue(is_in_memory(x1))
        self.assertEqual(allocated_memory(device), 4 * 1024 * 1024)
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
        self.assertTrue(is_in_memory(x4))
        x6 = x5 + 2
        self.assertFalse(is_in_memory(x2))
        # the eviction of x2 increases the cost of x3, so x4 is evicted
        self.assertTrue(is_in_memory(x3))
        self.assertFalse(is_in_memory(x4))

        self.assertTrue(np.array_equal(x6.numpy(), np.ones(x6.shape) * 11))
        self.assertTrue(np.array_equal(x3.numpy(), np.ones(x3.shape) * 5))

    @flow.unittest.skip_unless_1n1d()
    @memory_budget(12, "cpu")
    def test_remat_full_and_init_constant(self, device):
        x1 = flow.eye(1024, 1024, device=device)
        self.assertTrue(is_in_memory(x1))
        self.assertEqual(allocated_memory(device), 4 * 1024 * 1024)

        x2 = flow.full(x1.shape, 3.0, device=device)
        flow.nn.init.constant_(x1, x2)  # type: ignore[arg-type]
        del x2
        self.assertEqual(allocated_memory(device), 4 * 1024 * 1024)

        evict(x1)

        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape) * 3))

    @flow.unittest.skip_unless_1n1d()
    @memory_budget(12, "cpu")
    def test_remat_lifecycle_of_view_tensor(self, device):
        x1 = flow.eye(2, 3, device=device)
        self.assertTrue(is_in_memory(x1))

        x2 = flow.ones(3, device=device)
        x3 = flow.expand(x2, (2, 3))
        x1[:] = x3
        del x3
        del x2

        evict(x1)

        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape)))

    @flow.unittest.skip_unless_1n1d()
    @memory_budget(16, "cpu")
    def test_remat_init_constant_and_scalar(self, device):
        x0 = flow.ones(1024, 1024).to(device)
        x1 = x0 + 0
        x2 = x1 + 1
        flow.nn.init.constant_(x1, 5.0)  # type: ignore[arg-type]

        evict(x1)
        self.assertTrue(np.array_equal(x1.numpy(), np.ones(x1.shape) * 5))

        evict(x1)
        evict(x2)
        self.assertTrue(np.array_equal(x2.numpy(), np.ones(x2.shape) * 2))

    @flow.unittest.skip_unless_1n1d()
    @memory_budget(80, "cpu")
    def test_copy(self, device):
        x1 = flow.ones(1)
        x2 = x1.to(device)
        self.assertTrue(x2.device.rematable)
        x3 = x2.to(flow.int64)
        self.assertTrue(x3.device.rematable)
        x4 = x2 + 1
        self.assertTrue(x4.device.rematable)

    @flow.unittest.skip_unless_1n1d()
    @memory_budget(80, "cuda")
    def test_simple_network(self, device):
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ).to(device)
        for p in model.parameters():
            p.grad = flow.zeros_like(p).to(device)
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        x = flow.ones(4, 3, 224, 224).to(device)
        mem = allocated_memory(device)
        for _ in range(10):
            mem2 = allocated_memory(device)
            self.assertEqual(mem, mem2)
            loss = model(x).sum()
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()

    def _test_resnet18(self, optimizer_fn, mode, expected_loss):
        flow.manual_seed(flow.env.get_rank())
        def to_remat(x):
            if mode == 'global':
                return x.to_global(flow.placement.all("cpu+remat"), flow.sbp.broadcast)
            else:
                return x.to('cpu+remat')

        model = to_remat(flowvision.models.resnet18())
        if mode == 'ddp':
            model = flow.nn.parallel.DistributedDataParallel(model, use_bucket=False)
        criterion = to_remat(nn.CrossEntropyLoss())

        for x in model.parameters():
            x.grad = to_remat(flow.zeros_like(x))
        # optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        optimizer = optimizer_fn(model.parameters())
        x = to_remat(flow.rand(10, 3, 224, 224))
        target = (
            to_remat(flow.randint(low=0, high=1000, size=(x.shape[0],)).to(flow.int32))
        )
        # NOTE: there is a bug in current implementation about random ops:
        # x1 = flow.rand(5)
        # x2 = x1 + 1
        # del x1   <--- we cannot block the eviction of x1 here because it is controlled by the user
        # evict(x2)
        # recompute(x2) <-- recomputing x2 triggers the recomputation of x1 and causes inconsistentness
        print(x.placement)
        print(x.to_local().device)
        flow._oneflow_internal.remat.disable_eviction(x)
        flow._oneflow_internal.remat.disable_eviction(target)
        ITER_NUM = 5
        for i in range(ITER_NUM):
            print("start allocated_memory(cpu):", allocated_memory("cpu"))
            print(
                "recomputation num: ", flow._oneflow_internal.remat.recomputation_num()
            )
            output = model(x)
            loss = criterion(output, target)
            del output
            print(loss.numpy().item())
            if i == 4 and expected_loss is not None:
                self.assertTrue(loss.numpy().item() in expected_loss)
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()
            print("end allocated_memory(cpu):", allocated_memory("cpu"))
            print(
                "recomputation num: ", flow._oneflow_internal.remat.recomputation_num()
            )

        # check there is more than 10 recomputations each iteration
        # so the correctness check makes sense.
        self.assertGreater(
            flow._oneflow_internal.remat.recomputation_num(), ITER_NUM * 10
        )

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(220, "cpu")
    @loss_test()
    def test_resnet18_naive_sgd(self, _):
        # NOTE: this loss is only correct in my environment on 21
        self._test_resnet18(
            lambda params: flow.optim.SGD(params, lr=0.1, momentum=0),
            'naive',
            [0.6304041147232056],
        )

    @flow.unittest.skip_unless_1n2d()
    @only_fbip()
    @memory_budget(220, "cpu")
    @loss_test()
    def test_resnet18_naive_sgd_ddp_1n2d(self, _):
        # 2 devices, 2 losses
        # NOTE: these losses are only correct in my environment on 21
        self._test_resnet18(
            lambda params: flow.optim.SGD(params, lr=0.1, momentum=0),
            'ddp',
            [1.8890058994293213, 1.8992782831192017],
        )

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(270, "cpu")
    @loss_test()
    def test_resnet18_momentum_sgd(self, _):
        # NOTE: this loss is only correct in my environment on 21
        self._test_resnet18(
            lambda params: flow.optim.SGD(params, lr=0.1, momentum=0.9), 'naive', None
        )

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(310, "cpu")
    @loss_test()
    def test_resnet18_adam(self, _):
        # NOTE: this loss is only correct in my environment on 21
        self._test_resnet18(lambda params: flow.optim.Adam(params, lr=0.1), 'naive', None)

    @flow.unittest.skip_unless_1n1d()
    @only_fbip()
    @memory_budget(410, "cpu")
    @loss_test()
    def test_resnet18_global_adam(self, _):
        # NOTE: this loss is only correct in my environment on 21
        self._test_resnet18(lambda params: flow.optim.Adam(params, lr=0.1), 'global', None)

    @flow.unittest.skip_unless_1n1d()
    @only_copy_on_write()
    @memory_budget(12, "cpu")
    def test_copy_on_write(self, _):
        x1 = flow.ones(1024 * 1024)  # 4MB
        x2 = flow.ones(1024 * 1024)
        x3 = x2 + 1
        x2 += x1
        display("cpu")
        print(f"x1 in memory?: {is_in_memory(x1)}")
        print(f"x2 in memory?: {is_in_memory(x2)}")
        print(f"x3 in memory?: {is_in_memory(x3)}")

        print(f"recompute num: {flow._oneflow_internal.remat.recomputation_num()}")
        print(
            f"forced eviction num: {flow._oneflow_internal.remat.forced_eviction_num()}"
        )
        print(
            f"eager eviction num: {flow._oneflow_internal.remat.eager_eviction_num()}"
        )

        print("-------------")

        print(x3.numpy())
        print(f"x1 in memory?: {is_in_memory(x1)}")
        print(f"x2 in memory?: {is_in_memory(x2)}")
        print(f"x3 in memory?: {is_in_memory(x3)}")

        print(f"recompute num: {flow._oneflow_internal.remat.recomputation_num()}")
        print(
            f"forced eviction num: {flow._oneflow_internal.remat.forced_eviction_num()}"
        )
        print(
            f"eager eviction num: {flow._oneflow_internal.remat.eager_eviction_num()}"
        )


if __name__ == "__main__":
    unittest.main()
