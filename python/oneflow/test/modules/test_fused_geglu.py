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
import numpy as np
from collections import OrderedDict

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

# from oneflow.test_utils.automated_test_util import *


class Geglu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: flow.Tensor, w: flow.Tensor, b: flow.Tensor) -> flow.Tensor:
        # matmul
        temp_matmul = flow._C.matmul(
            input=x, other=w, transpose_a=False, transpose_b=True
        )

        # add bias
        temp_add_bias = flow._C.add(input=temp_matmul, other=b)

        # chunk
        hidden_state, gate = temp_add_bias.chunk(2, dim=-1).contiguous()  # sync

        # gelu and element-wise product
        return hidden_state * flow.gelu(gate)


def _test_fused_geglu_profile(test_case, params: dict):
    # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]
    input = np.random.randn(m, k)
    weight = np.random.randn(k, n * 2)
    weight_t = np.transpose(weight)
    bias = np.random.randn(n * 2)

    # test: fused op
    flow_input_tensor = flow.Tensor(input).to("cuda")
    flow_weight_t_tensor = flow.Tensor(weight_t).to("cuda")
    flow_bias_tensor = flow.Tensor(bias).to("cuda")
    fused_out = flow._C.fused_geglu(
        flow_input_tensor, flow_weight_t_tensor, flow_bias_tensor
    )


def _test_fused_geglu(test_case, params: dict):
    # config test data
    m = params["m"]
    n = params["n"]
    k = params["k"]
    input = np.random.randn(m, k)
    weight = np.random.randn(k, n * 2)
    weight_t = np.transpose(weight).contiguous()  # sync
    bias = np.random.randn(n * 2)

    # test: fused op
    flow_input_tensor = flow.Tensor(input).to("cuda")
    flow_weight_t_tensor = flow.Tensor(weight_t).to("cuda")
    flow_bias_tensor = flow.Tensor(bias).to("cuda")
    fused_out = flow._C.fused_geglu(
        flow_input_tensor, flow_weight_t_tensor, flow_bias_tensor
    )

    # test: naive result
    stack_bias = np.stack((bias,) * m, axis=0)
    flow_stack_bias_tensor = flow.Tensor(stack_bias).to("cuda")
    flow_module = Geglu()
    origin_out = flow_module.forward(
        x=flow_input_tensor, w=flow_weight_t_tensor, b=flow_stack_bias_tensor
    )

    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedGeglu(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_fused_geglu,
            # _test_fused_geglu_profile
        ]
        arg_dict["params"] = [
            {"m": 256, "k": 1280, "n": 5120},
            {"m": 1024, "k": 640, "n": 2560},
            {"m": 4096, "k": 320, "n": 1280},
        ]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
            time.sleep(2)


if __name__ == "__main__":
    unittest.main()
