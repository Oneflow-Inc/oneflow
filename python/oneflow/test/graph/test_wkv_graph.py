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
import oneflow.nn as nn
import oneflow.unittest


def test_wkv_graph(B, T, C, w, u, k, v):
    model = nn.Sequential(nn.Linear(1, 1, bias=True)).to("cuda")
    optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss().to("cuda")

    class WkvGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()
            self.model = model
            self.loss_fn = loss_fn
            self.add_optimizer(optimizer)
            self.config.enable_amp(True)

        def build(self, x, w, u, k, v):
            w = self.model(x).squeeze(0) + w
            y = flow._C.wkv(B, T, C, w, u, k, v)
            loss = y.sum()
            print(loss)
            loss.backward()
            return y

    model = WkvGraph()
    # model.debug(2, op_repr_with_py_stack=True, only_user_py_stack=False, max_py_stack_depth=10)
    x = flow.randn(1, 1).to("cuda")
    return model(x, w, u, k, v)


def CHECK_CUDA():
    B = 64
    T = 256
    C = 128

    w = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
    u = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
    k = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
    v = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)

    test_wkv_graph(B, T, C, w, u, k, v)


if __name__ == "__main__":
    CHECK_CUDA()
