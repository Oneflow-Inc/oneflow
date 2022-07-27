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
import oneflow as torch
from math import exp


def RUN_FORMULA_VERY_SLOW(B, T, C, w, u, k, v):
    w = -torch.exp(w)
    out = torch.empty((B, T, C), device="cuda")
    for b in range(B):
        for c in range(C):
            out[b][0][c] = v[b][0][c]
            for t in range(1, T):
                p = 0
                q = 0
                for s in range(t + 1):
                    if s == t:
                        ek = exp(k[b][s][c] + u[c])
                    else:
                        ek = exp(k[b][s][c] + w[c] * (t - s - 1))
                    p += ek * v[b][s][c]
                    q += ek
                out[b][t][c] = p / q
    return out


wkv_cuda = torch._C


class WKV:
    def forward(B, T, C):
        w = torch.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        w = -torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        y = torch.empty((B, T, C), device="cuda")
        wkv_cuda.wkv_forward(B, T, C, w, u, k, v, y)
        return y

    def backward(B, T, C):
        w = torch.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device="cuda").uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        gy = torch.zeros(B, T, C, requires_grad=True, device="cuda").uniform_(-1, 1)
        gw = torch.zeros((B, C), device="cuda")
        gu = torch.zeros((B, C), device="cuda")
        gk = torch.zeros((B, T, C), device="cuda")
        gv = torch.zeros((B, T, C), device="cuda")
        wkv_cuda.wkv_backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        return (None, None, None, gw, gu, gk, gv)


def CHECK_CUDA():
    B = 32
    T = 768
    C = 768

    # y = WKV.forward(B, T, C)
    WKV.backward(B, T, C)


if __name__ == "__main__":
    CHECK_CUDA()
