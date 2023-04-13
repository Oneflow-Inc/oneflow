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


@contextmanager
def generate_placeholder(size_mb, device):
    global placeholder_size
    placeholder_size = size_mb * 1024 * 1024
    print(placeholder_size)
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

class TestGraphRemat(flow.unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flow.remat.set_budget("500MB")
        flow.remat.set_small_pieces_optimization(False)

    # test basic remat comparing loss and remat num with graph and eager
    @memory_budget(12, "cpu")
    def test_graph_basic_remat(self,device):
  
        loss_eager=[]
        loss_graph=[]
        class basicModule(nn.Module):
            def __init__(self):
                super().__init__()
        
            def forward(self, input):
                x1 = input*-2
                x2 = x1-2
                x3= x2*0.1
                return x3

        model=basicModule()
        class GraphMy(nn.Graph):
            def __init__(self):
                super().__init__()
                self.model=model
            def build(self,x):
                out=self.model(x)
                return out
        graph=GraphMy()
        input = flow.ones(1024*1024, device=device) # 4MB
        for _ in range(10):
            loss=model(input).sum()
            del loss
        num_eager=flow._oneflow_internal.remat.recomputation_num()
        for _ in range(10):
            loss=graph(input).sum()
            del loss
        num_graph=flow._oneflow_internal.remat.recomputation_num()-num_eager-1
        self.assertEqual(num_eager,num_graph)


if __name__ == "__main__":
    unittest.main()
