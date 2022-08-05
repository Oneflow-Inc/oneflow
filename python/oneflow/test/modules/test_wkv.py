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
import unittest
import random

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestWkv(unittest.TestCase):
    def test_wkv(test_case):
        B = random.randint(1, 10)
        T = random.randint(1, 10)
        C = random.randint(1, 10)
        w = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        u = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        k = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        v = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        y = flow._C.wkv(B, T, C, w, u, k, v)
        y.sum().backward()
        print(w.shape)
        print(w.grad.shape)
        return y

    def test_graph_wkv(test_case):
        B = random.randint(1, 10)
        T = random.randint(1, 10)
        C = random.randint(1, 10)
        w = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        u = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        k = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        v = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)

        class WkvGraph(flow.nn.Graph):
            def __init__(self,):
                super().__init__()

            def build(self, w, u, k, v):
                y = flow._C.wkv(B, T, C, w, u, k, v)
                return y

        model = WkvGraph()
        return model(w, u, k, v)


if __name__ == "__main__":
    unittest.main()
