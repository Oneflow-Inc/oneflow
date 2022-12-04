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

is_profiling = False


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
        matmul_wx = flow._C.matmul(
            input=x, other=w, transpose_a=False, transpose_b=True
        )
        if split_mode:
            matmul_vx = flow._C.matmul(
                input=x, other=v, transpose_a=False, transpose_b=True
            )

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


"""
    @desp: profiling fused glu implementation with split weight matrix
"""


def _test_fused_glu_split_profiling(test_case, params: dict):
    # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]
    act = params["act"]

    # generate random input
    x = np.random.randn(2, m, k)
    w = np.random.randn(n, k)  # transpose
    # w_t = np.transpose(weight).contiguous() #sync
    b = np.random.randn(n)
    v = np.random.randn(n, k)  # transpose
    c = np.random.randn(n)

    # transfer to gpu memory
    input_tensor_x = flow.FloatTensor(x).to("cuda")
    input_tensor_w = flow.FloatTensor(w).to("cuda")
    input_tensor_b = flow.FloatTensor(b).to("cuda")
    input_tensor_v = flow.FloatTensor(v).to("cuda")
    input_tensor_c = flow.FloatTensor(c).to("cuda")

    # test: fused op
    output_tensor_y = flow._C.fused_glu(
        x=input_tensor_x,
        w=input_tensor_w,
        b=input_tensor_b,
        v=input_tensor_v,
        c=input_tensor_c,
        activation=act,
    )


"""
    @desp: check the functionality of fused glu implementation with split weight matrix
"""


def _test_fused_glu_split(test_case, params: dict):
    # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]
    act = params["act"]

    # generate random input
    x = np.random.randn(2, m, k)
    w = np.random.randn(n, k)  # transpose
    # w_t = np.transpose(weight).contiguous() #sync
    b = np.random.randn(n)
    v = np.random.randn(n, k)  # transpose
    c = np.random.randn(n)

    # transfer to gpu memory
    input_tensor_x = flow.FloatTensor(x).to("cuda")
    input_tensor_w = flow.FloatTensor(w).to("cuda")
    input_tensor_b = flow.FloatTensor(b).to("cuda")
    input_tensor_v = flow.FloatTensor(v).to("cuda")
    input_tensor_c = flow.FloatTensor(c).to("cuda")

    # test: fused op
    output_tensor_y = flow._C.fused_glu(
        x=input_tensor_x,
        w=input_tensor_w,
        b=input_tensor_b,
        v=input_tensor_v,
        c=input_tensor_c,
        activation=act,
    )

    # test: naive result
    flow_module = Glu()
    origin_output_tensor_y = flow_module.forward(
        x=input_tensor_x,
        w=input_tensor_w,
        b=input_tensor_b,
        v=input_tensor_v,
        c=input_tensor_c,
        split_mode=True,
        activation=act,
    )

    test_case.assertTrue(
        np.allclose(
            output_tensor_y.numpy(),
            origin_output_tensor_y.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


"""
    @desp: profiling fused glu implementation
"""


def _test_fused_glu_profiling(test_case, params: dict):
    # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]
    act = params["act"]

    # generate random input
    x = np.random.randn(2, m, k)
    w = np.random.randn(n * 2, k)  # transpose
    # w_t = np.transpose(weight).contiguous() #sync
    b = np.random.randn(n * 2)

    # transfer tensors to gpu memory
    input_tensor_x = flow.FloatTensor(x).to("cuda")
    input_tensor_w = flow.FloatTensor(w).to("cuda")
    input_tensor_b = flow.FloatTensor(b).to("cuda")

    # test: fused op
    output_tensor_y = flow._C.fused_glu(
        x=input_tensor_x,
        w=input_tensor_w,
        b=input_tensor_b,
        v=None,
        c=None,
        activation=act,
    )


"""
    @desp: check the functionality of fused glu implementation
"""


def _test_fused_glu(test_case, params: dict):
    # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]
    act = params["act"]

    # generate random input
    x = np.random.randn(2, m, k)
    w = np.random.randn(n * 2, k)  # transpose
    # w_t = np.transpose(weight).contiguous() #sync
    b = np.random.randn(n * 2)

    # transfer tensors to gpu memory
    input_tensor_x = flow.FloatTensor(x).to("cuda")
    input_tensor_w = flow.FloatTensor(w).to("cuda")
    input_tensor_b = flow.FloatTensor(b).to("cuda")

    # test: fused op
    output_tensor_y = flow._C.fused_glu(
        x=input_tensor_x,
        w=input_tensor_w,
        b=input_tensor_b,
        v=None,
        c=None,
        activation=act,
    )

    # test: naive result
    flow_module = Glu()
    origin_output_tensor_y = flow_module.forward(
        x=input_tensor_x,
        w=input_tensor_w,
        b=input_tensor_b,
        split_mode=False,
        activation=act,
    )

    test_case.assertTrue(
        np.allclose(
            output_tensor_y.numpy(),
            origin_output_tensor_y.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedGlu(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        # set up test functions
        if is_profiling:
            # for profiling test
            arg_dict["test_fun"] = [
                _test_fused_glu_profiling,
                _test_fused_glu_split_profiling,
            ]
        else:
            # for functionality test
            arg_dict["test_fun"] = [
                _test_fused_glu,
            ]

        # set up profiling functions
        if is_profiling:
            # set up test functions
            arg_dict["params"] = [
                # for profiling
                {"m": 256, "k": 1280, "n": 1280, "act": "none"},
                {"m": 256, "k": 1280, "n": 2560, "act": "none"},
                {"m": 256, "k": 1280, "n": 5120, "act": "none"},
            ]
        else:
            # for functionality test
            arg_dict["params"] = [
                # m=256, k=1280, n=5120
                {"m": 256, "k": 1280, "n": 5120, "act": "fast_gelu"},
                {"m": 1024, "k": 640, "n": 2560, "act": "fast_gelu"},
                {"m": 4096, "k": 320, "n": 1280, "act": "fast_gelu"},
            ]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
