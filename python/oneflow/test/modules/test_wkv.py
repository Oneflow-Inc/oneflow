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


class WKV:
    def forward(B, T, C):
        w = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        u = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        k = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        v = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        w = -flow.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        y = flow.empty((B, T, C), device="cuda")
        flow._C.wkv_forward(B, T, C, w, u, k, v, y)
        return y

    def backward(B, T, C):
        w = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        u = flow.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        k = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        v = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        gy = flow.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        gw = flow.zeros((B, C), device="cuda")
        gu = flow.zeros((B, C), device="cuda")
        gk = flow.zeros((B, T, C), device="cuda")
        gv = flow.zeros((B, T, C), device="cuda")
        flow._C.wkv_grad(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        return (None, None, None, gw, gu, gk, gv)


@flow.unittest.skip_unless_1n1d()
class TestWkv(unittest.TestCase):
    def test_wkv(test_case):
        B = random.randint(1, 10)
        T = random.randint(1, 10)
        C = random.randint(1, 10)

        WKV.forward(B, T, C)
        WKV.backward(B, T, C)


if __name__ == "__main__":
    unittest.main()
