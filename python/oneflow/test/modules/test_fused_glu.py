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
import os
import unittest
import time
import datetime
import numpy as np
from collections import OrderedDict

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

class Glu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: flow.Tensor,
        w: flow.Tensor,
        b: flow.Tensor,
        v: flow.Tensor = None,
        c: flow.Tensor = None,
        split_mode: bool = False,
        activation: str = "none",
    ) -> flow.Tensor:
        # matmul
        matmul_wx = flow._C.matmul(input=x, other=w, transpose_a=False, transpose_b=True)
        if split_mode:
            matmul_vx = flow._C.matmul(input=x, other=v, transpose_a=False, transpose_b=True)

        # add bias
        matmul_wx_b = flow._C.add(input=matmul_wx, other=b)
        if split_mode:
            matmul_vx_c = flow._C.add(input=matmul_vx, other=c)

        # chunk
        if split_mode:
            hidden_state = matmul_wx_b
            gate = matmul_vx_c
        else:
            hidden_state, gate = matmul_wx_b.chunk(2, dim=-1)

        # activation and element-wise product
        if activation == "none":
            return hidden_state * gate
        elif activation == "sigmoid":
            return hidden_state * flow.sigmoid(gate)
        elif activation == "relu":
            return hidden_state * flow.relu(gate)
        elif activation == "gelu":
            return hidden_state * flow.gelu(gate)
        elif activation == "fast_gelu":
            return hidden_state * flow._C.fast_gelu(gate)
        elif activation == "silu":
            return hidden_state * flow.silu(gate)

def tensor_builder(params: dict, dtype=flow.float32, is_split_mode=True):
     # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]

    # generate random input
    x = np.random.randn(2, m, k)
    if is_split_mode:
        w = np.random.randn(n, k)  # transpose
        b = np.random.randn(n)
        v = np.random.randn(n, k)  # transpose
        c = np.random.randn(n)
    else:
        w = np.random.randn(n * 2, k)  # transpose
        b = np.random.randn(n * 2)

    # transfer to gpu memory
    tensor_x = flow.FloatTensor(x).to(dtype=dtype, device="cuda")
    tensor_w = flow.FloatTensor(w).to(dtype=dtype, device="cuda").requires_grad_(True)
    tensor_b = flow.FloatTensor(b).to(dtype=dtype, device="cuda").requires_grad_(True)
    if is_split_mode:
        tensor_v = flow.FloatTensor(v).to(dtype=dtype, device="cuda").requires_grad_(True)
        tensor_c = flow.FloatTensor(c).to(dtype=dtype, device="cuda").requires_grad_(True)

    if is_split_mode:
        return tensor_x, tensor_w, tensor_b, tensor_v, tensor_c
    else:
        return tensor_x, tensor_w, tensor_b

def compare_result(test_case, a, b, rtol=1e-5, atol=1e-8):
    test_case.assertTrue(
        np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
        f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
    )

def _test_fused_glu(test_case, params: dict, dtype=flow.float32):
    flow_module = Glu()
    x, w, b = tensor_builder(params=params, dtype=dtype, is_split_mode=False)

    y = flow_module.forward(x=x, w=w, b=b, split_mode=False, activation=params["act"])
    y.mean().backward()
    w_grad = w.grad.detach().cpu()
    b_grad = b.grad.detach().cpu()
    y = y.detach().cpu()

    fused_x = x.detach().clone()
    fused_w = w.detach().clone().requires_grad_(True)
    fused_b = b.detach().clone().requires_grad_(True)
    fused_y = flow._C.fused_glu(x=fused_x, w=fused_w, b=fused_b, v=None, c=None, activation=params["act"])
    fused_y.mean().backward()
    fused_w_grad = fused_w.grad.detach().cpu()
    fused_b_grad = fused_b.grad.detach().cpu()
    fused_y = fused_y.detach().cpu()

    if dtype == flow.float16:
        compare_result(test_case, fused_y, y, 1e-2, 1e-3)
        compare_result(test_case, fused_w_grad, w_grad, 1e-4, 1e-3)
        compare_result(test_case, fused_b_grad, b_grad, 1e-4, 1e-3)
    else:
        compare_result(test_case, fused_y, y)
        compare_result(test_case, fused_w_grad, w_grad)
        compare_result(test_case, fused_b_grad, b_grad)
    print(str(dtype) + ", " + params["act"] + " passed!" )

def _test_fused_glu_split(test_case, params: dict, dtype=flow.float32):
    flow_module = Glu()
    x, w, b, v, c = tensor_builder(params=params, dtype=dtype, is_split_mode=True)

    y = flow_module.forward(x=x, w=w, b=b, v=v, c=c, split_mode=True, activation=params["act"])
    y.mean().backward()
    w_grad = w.grad.detach().cpu()
    b_grad = b.grad.detach().cpu()
    v_grad = v.grad.detach().cpu()
    c_grad = c.grad.detach().cpu()
    y = y.detach().cpu()

    fused_x = x.detach().clone()
    fused_w = w.detach().clone().requires_grad_(True)
    fused_b = b.detach().clone().requires_grad_(True)
    fused_v = v.detach().clone().requires_grad_(True)
    fused_c = c.detach().clone().requires_grad_(True)
    fused_y = flow._C.fused_glu(x=fused_x, w=fused_w, b=fused_b, v=fused_v, c=fused_c, activation=params["act"])
    fused_y.mean().backward()
    fused_w_grad = fused_w.grad.detach().cpu()
    fused_b_grad = fused_b.grad.detach().cpu()
    fused_v_grad = fused_v.grad.detach().cpu()
    fused_c_grad = fused_c.grad.detach().cpu()
    fused_y = fused_y.detach().cpu()

    if dtype == flow.float16:
        compare_result(test_case, fused_y, y, 1e-2, 1e-3)
        compare_result(test_case, fused_w_grad, w_grad, 1e-4, 1e-3)
        compare_result(test_case, fused_b_grad, b_grad, 1e-4, 1e-3)
        compare_result(test_case, fused_v_grad, v_grad, 1e-4, 1e-3)
        compare_result(test_case, fused_c_grad, c_grad, 1e-4, 1e-3)
    else:
        compare_result(test_case, fused_y, y)
        compare_result(test_case, fused_w_grad, w_grad)
        compare_result(test_case, fused_b_grad, b_grad)
        compare_result(test_case, fused_v_grad, v_grad)
        compare_result(test_case, fused_c_grad, c_grad)
    print(str(dtype) + ", " + params["act"] + " passed!" )

@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedGlu(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        # set up test functions
        arg_dict["test_fun"] = [
            _test_fused_glu,
            _test_fused_glu_split,
        ]

        # set up profiling functions
        arg_dict["params"] = [
            # m=256, k=1280, n=5120
            {"m": 256, "k": 1280, "n": 5120, "act": "none"},
            {"m": 256, "k": 1280, "n": 5120, "act": "sigmoid"},
            {"m": 256, "k": 1280, "n": 5120, "act": "relu"},
            {"m": 256, "k": 1280, "n": 5120, "act": "gelu"},
            {"m": 256, "k": 1280, "n": 5120, "act": "fast_gelu"},
            {"m": 256, "k": 1280, "n": 5120, "act": "silu"},
            # m=1024, k=640, n=2560
            {"m": 1024, "k": 640, "n": 2560, "act": "none"},
            {"m": 1024, "k": 640, "n": 2560, "act": "sigmoid"},
            {"m": 1024, "k": 640, "n": 2560, "act": "relu"},
            {"m": 1024, "k": 640, "n": 2560, "act": "gelu"},
            {"m": 1024, "k": 640, "n": 2560, "act": "fast_gelu"},
            {"m": 1024, "k": 640, "n": 2560, "act": "silu"},
            # m=4096, k=320, n=1280
            {"m": 4096, "k": 320, "n": 1280, "act": "none"},
            {"m": 4096, "k": 320, "n": 1280, "act": "sigmoid"},
            {"m": 4096, "k": 320, "n": 1280, "act": "relu"},
            {"m": 4096, "k": 320, "n": 1280, "act": "gelu"},
            {"m": 4096, "k": 320, "n": 1280, "act": "fast_gelu"},
            {"m": 4096, "k": 320, "n": 1280, "act": "silu"},
        ]

        arg_dict["dtype"] = [flow.float16, flow.float32]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
