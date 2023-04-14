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


class NanoGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, x):
        out = self.model(x)
        return out


class SubGraph(nn.Graph):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.add_optimizer(optimizer)

    def build(self, input):
        loss = self.model(input).sum()
        loss.backward()
        return loss


def test_remat_forward_backward(model, graph, input, task, B=False):
    loss_graph_remat = []
    loss_graph_ori = []
    num_eager, num_graph = 0, 0
    if task == "num":
        for _ in range(10):
            loss = model(input).sum()
            if B == True:
                loss.backward()
            del loss
        num_eager = get_num_remat()

    if task == "loss":
        device = "cpu"
        graph = NanoGraph(model)
        input = flow.ones(1024*1024, device=device)
        if B == True:
            input = flow.rand(4, 3, 1024, 1024).to(device)
        for _ in range(10):
            loss = graph(input).sum()
            loss_graph_ori.append(loss.numpy())
            del loss

    for _ in range(10):
        loss = graph(input).sum()
        if task == "loss":
            loss_graph_remat.append(loss.numpy())
        del loss
    if task == "num":
        num_graph = get_num_remat()

    return num_eager, num_graph, loss_graph_remat, loss_graph_ori


class TestGraphRemat(flow.unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flow.remat.set_budget("2000MB")
        flow.remat.set_small_pieces_optimization(False)

    # test remat num in nano module graph and eager
    @memory_budget(12, "cpu")
    def test_graph_basic_remat(self, device):
        input = flow.ones(1024*1024, device=device)  # 4MB

        model = NanoModule().to(device)
        graph = NanoGraph(model)

        num_eager, num_graph, _, _ = test_remat_forward_backward(
            model, graph, input, "num")

        self.assertEqual(num_eager, num_graph-1)

    # test loss equality  with rematable graph and unrematable graph
    @memory_budget(12, "cpu")
    def test_graph_basic_remat2(self, device):
        input = flow.ones(1024*1024, device=device)

        model = NanoModule().to(device)
        graph = NanoGraph(model)
        _, _, loss_graph_remat, loss_graph_ori = test_remat_forward_backward(
            model, graph, input, "loss")

        self.assertEqual(loss_graph_remat, loss_graph_ori)

    # test backward advanced remat comparing loss with rematable graph and unrematable graph
    def test_graph_backward_remat(self):
        device = "cpu+remat"
        flow.remat.set_budget("500MB")

        flow.manual_seed(1213)
        model = SubModule().to(device)
        for p in model.parameters():
            p.grad = flow.zeros_like(p).to(device)
        optimizer = flow.optim.SGD(model.parameters(), lr=0.1, momentum=0)
        graph = SubGraph(model, optimizer)

        input = flow.rand(4, 3, 1024, 1024).to(device)

        num_eager, num_graph, _, _ = test_remat_forward_backward(
            model, graph, input, "num", B=True)
        self.assertEqual(num_eager, num_graph)

        model = SubModule().to("cpu")
        _, _, loss_graph_remat, loss_graph_ori = test_remat_forward_backward(
            model, graph, input, "loss", B=True)

        self.assertEqual(loss_graph_remat, loss_graph_ori)


if __name__ == "__main__":
    unittest.main()
