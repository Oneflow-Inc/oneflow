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


NUM_REMAT = []


def get_num_remat():
    NUM_REMAT.append(flow._oneflow_internal.remat.recomputation_num())
    if len(NUM_REMAT) == 1:
        return NUM_REMAT[0]
    return NUM_REMAT[-1]-NUM_REMAT[-2]


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


class NanoModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x1 = input*-2
        x2 = x1-2
        x3 = x2*0.1
        return x3


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, 2, 1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        x1 = self.conv2d(input)
        x2 = self.bn(x1)
        x3 = self.relu(x2)
        return x3


class TestGraphRemat(flow.unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flow.remat.set_budget("2000MB")
        flow.remat.set_small_pieces_optimization(False)

    # test basic remat comparing loss and remat num with graph and eager
    @memory_budget(12, "cpu")
    def test_graph_basic_remat(self, device):

        class basicModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                x1 = input*-2
                x2 = x1-2
                x3 = x2*0.1
                return x3

        model = basicModule().to(device)

        class GraphMy(nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model

            def build(self, x):
                out = self.model(x)
                return out
        graph = GraphMy()
        input = flow.ones(1024*1024, device=device)  # 4MB
        for _ in range(10):
            loss = model(input).sum()
            del loss
        num_eager = get_num_remat()

        for _ in range(10):
            loss = graph(input).sum()
            del loss

        num_graph = get_num_remat()
        self.assertEqual(num_eager, num_graph-1)

    # test basic remat comparing loss with rematable graph and unrematable graph
    @memory_budget(12, "cpu")
    def test_graph_basic_remat2(self, device):
        loss_graph_remat = []
        loss_graph_ori = []

        class basicModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                x1 = input*-2
                x2 = x1-2
                x3 = x2*0.1
                return x3

        model = basicModule().to(device)

        class GraphMy(nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model

            def build(self, x):
                out = self.model(x)
                return out
        graph = GraphMy()
        input = flow.ones(1024*1024, device=device)

        for _ in range(10):
            loss = graph(input).sum()
            loss_graph_remat.append(loss.numpy())
            del loss
        get_num_remat()

        # graph with no remat
        device = "cpu"
        input = flow.ones(1024*1024, device=device)
        for _ in range(10):
            loss = graph(input).sum()
            loss_graph_ori.append(loss.numpy())
            del loss
        get_num_remat()
        self.assertEqual(loss_graph_remat, loss_graph_ori)

    # test backward advanced remat comparing loss with rematable graph and unrematable graph
    def test_graph_backward_remat(self):
        device = "cpu+remat"
        flow.remat.set_budget("500MB")
        loss_graph_remat = []
        loss_graph_ori = []

        class ModuleMyAdvanced(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = nn.Conv2d(3, 32, 3, 2, 1)
                self.bn = nn.BatchNorm2d(32)
                self.relu = nn.ReLU(inplace=False)

            def forward(self, input):
                x1 = self.conv2d(input)
                x2 = self.bn(x1)
                x3 = self.relu(x2)
                return x3

        class GraphMyBackward(nn.Graph):
            def __init__(self,):
                super().__init__()
                self.model = model
                self.add_optimizer(optimizer)

            def build(self, input):
                loss = self.model(input).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                return loss

        flow.manual_seed(1213)
        model = ModuleMyAdvanced().to(device)
        for p in model.parameters():
            p.grad = flow.zeros_like(p).to(device)
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        input = flow.rand(4, 3, 1024, 1024).to(device)
        graph = GraphMyBackward()
        for _ in range(10):
            loss = graph(input).sum()
            loss_graph_remat.append(loss.numpy())
            del loss
        num_graph = get_num_remat()

        # eager with remat
        flow.manual_seed(1213)
        device = "cpu+remat"
        model = ModuleMyAdvanced().to(device)
        for p in model.parameters():
            p.grad = flow.zeros_like(p).to(device)
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        input = flow.rand(4, 3, 1024, 1024).to(device)
        for _ in range(10):
            loss = model(input).sum()
            loss.backward()
            del loss
        num_eager = get_num_remat()

        # graph with no remat
        device = "cpu"
        flow.manual_seed(1213)
        model = ModuleMyAdvanced().to(device)
        for p in model.parameters():
            p.grad = flow.zeros_like(p).to(device)
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        input = flow.rand(4, 3, 1024, 1024).to(device)
        graph = GraphMyBackward()
        for _ in range(10):
            loss = graph(input.to(device)).sum()
            loss_graph_ori.append(loss.numpy())
            del loss

        self.assertEqual(loss_graph_remat, loss_graph_ori)

        self.assertEqual(num_eager, num_graph)


if __name__ == "__main__":
    unittest.main()
