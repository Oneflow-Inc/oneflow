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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: oneflow.broadcast_matmul
# CHECK-NOT: oneflow.fused_matmul_bias
# CHECK-NOT: oneflow.narrow
# CHECK: "oneflow.fused_glu"

import unittest
import numpy as np

import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.unittest
import oneflow.sysconfig


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(
        self, dim_in: int, dim_out: int,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class GraphToRun(flow.nn.Graph):
    def __init__(self, gelu_mod):
        super().__init__()
        self.gelu_mod = gelu_mod

    def build(self, hidden_states):
        return self.gelu_mod(hidden_states)


def do_fused_gelu_graph(test_case, dev, fuse_linear=False):
    if fuse_linear:
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
    else:
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "0"
    gelu_mod = GEGLU(640, 5120).to(dev)
    hidden_states = flow.randn(2, 2304, 640).to(dev)
    eager_res = gelu_mod(hidden_states)
    graph_to_run = GraphToRun(gelu_mod)
    lazy_res = graph_to_run(hidden_states)
    test_case.assertTrue(np.allclose(eager_res.numpy(), lazy_res.numpy()))


@flow.unittest.skip_unless_1n1d()
@unittest.skipUnless(oneflow.sysconfig.with_cuda(), "needs -DBUILD_CUDA=ON")
class TestFusedGelu(oneflow.unittest.TestCase):
    def test_fused_gelu_graph(test_case):
        do_fused_gelu_graph(test_case, "cuda", fuse_linear=True)
        do_fused_gelu_graph(test_case, "cuda", fuse_linear=False)


if __name__ == "__main__":
    unittest.main()
