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
import pickle
import numpy as np
import oneflow as flow
import oneflow.nn as nn
import torch
from torch.utils.cpp_extension import load

CUDA_KERNEL_VERSION = 2
wkv_cuda = load(
    name="wkv",
    sources=[
        "/home/zhangxiaoyu/RWKV-CUDA/wkv/cuda/wkv_op.cpp",
        f"/home/zhangxiaoyu/RWKV-CUDA/wkv/cuda/wkv_cuda_v{CUDA_KERNEL_VERSION}.cu",
    ],
    verbose=True,
    extra_cuda_cflags=["--use_fast_math", "--extra-device-vectorization"],
)

file = open("./input.pkl", "rb")
input_dict = pickle.load(file)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda")
        gu = torch.zeros((B, C), device="cuda")
        gk = torch.zeros((B, T, C), device="cuda")
        gv = torch.zeros((B, T, C), device="cuda")
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def CHECK_CUDA():
    B = input_dict["B"]
    T = input_dict["T"]
    C = input_dict["C"]
    print(B, T, C)

    with torch.no_grad():
        # w = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-1, 1)
        # u = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-1, 1)
        # k = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)
        # v = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)
        w = torch.from_numpy(input_dict["time_decay"]).to("cuda").requires_grad_()
        u = torch.from_numpy(input_dict["time_first"]).to("cuda").requires_grad_()
        k = torch.from_numpy(input_dict["k"]).to("cuda").requires_grad_()
        v = torch.from_numpy(input_dict["v"]).to("cuda").requires_grad_()

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1 = RUN_CUDA(B, T, C, -torch.exp(w.contiguous()), u, k, v)
    loss1 = y1.sum()
    print("loss1, ", loss1)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss1.backward()

    gw = w.grad
    gu = u.grad
    gk = k.grad
    gv = v.grad
    gw_torch = gw.detach().cpu().numpy()
    gu_torch = gu.detach().cpu().numpy()
    gk_torch = gk.detach().cpu().numpy()
    gv_torch = gv.detach().cpu().numpy()

    w = flow.tensor(w.detach().cpu().numpy(), requires_grad=True, device="cuda")
    u = flow.tensor(u.detach().cpu().numpy(), requires_grad=True, device="cuda")
    k = flow.tensor(k.detach().cpu().numpy(), requires_grad=True, device="cuda")
    v = flow.tensor(v.detach().cpu().numpy(), requires_grad=True, device="cuda")
    y2 = flow._C.wkv(B, T, C, -flow.exp(w.contiguous()), u, k, v).requires_grad_()
    loss2 = y2.sum()
    print("loss2, ", loss2)
    loss2.backward()

    gw = w.grad
    gu = u.grad
    gk = k.grad
    gv = v.grad
    gw_flow = gw.detach().cpu().numpy()
    gu_flow = gu.detach().cpu().numpy()
    gk_flow = gk.detach().cpu().numpy()
    gv_flow = gv.detach().cpu().numpy()

    print(np.allclose(gw_flow, gw_torch, atol=1e-4))
    print(np.allclose(gu_flow, gu_torch, atol=1e-4))
    print(np.allclose(gk_flow, gk_torch, atol=1e-4))
    print(np.allclose(gv_flow, gv_torch, atol=1e-4))
    print(gw_flow.flatten()[:5], -np.exp(gw_torch.flatten()[:5]))
    print(gu_flow.flatten()[:5], gu_torch.flatten()[:5])
    print(gk_flow.flatten()[:5], gk_torch.flatten()[:5])
    print(gv_flow.flatten()[:5], gv_torch.flatten()[:5])
    # print(gu_flow, gu_torch)


if __name__ == "__main__":
    CHECK_CUDA()
