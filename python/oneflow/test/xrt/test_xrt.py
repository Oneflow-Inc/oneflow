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
import os
from random import randint
from random import choice

import numpy as np

import oneflow as flow
import oneflow.unittest


def generate_graph(func):
    class Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, *args):
            return func(*args)

    return Graph()


# graph need to be a new graph, due to the limit of nn.Graph
def test_xrt_openvino(test_case, graph, input, ref_out, rtol=1e-3, atol=1e-4):
    if os.getenv("ONEFLOW_TEST_FORCE_OPENVINO") and not flow.sysconfig.with_openvino():
        test_case.assertTrue(False)

    if not flow.sysconfig.with_openvino():
        return

    graph.config.enable_openvino(True)
    if len(input) == 1:
        out = graph(input)
    else:
        out = graph(*input)

    if len(ref_out) == 1:
        test_case.assertTrue(
            np.allclose(ref_out[0].numpy(), out.numpy(), rtol=rtol, atol=atol)
        )
    else:
        test_case.assertTrue(len(ref_out) == len(out))
        for i in range(len(ref_out)):
            test_case.assertTrue(
                np.allclose(ref_out[i].numpy(), out[i].numpy(), rtol=rtol, atol=atol)
            )


# graph need to be a new graph, due to the limit of nn.Graph
def test_xrt_tensorrt(test_case, graph, input, ref_out, rtol=1e-3, atol=1e-4):
    if os.getenv("ONEFLOW_TEST_FORCE_TENSORRT") and not flow.sysconfig.with_tensorrt():
        test_case.assertTrue(False)

    if not flow.sysconfig.with_tensorrt():
        return

    graph.config.enable_tensorrt(True)
    if len(input) == 1:
        out = graph(input)
    else:
        out = graph(*input)

    if len(ref_out) == 1:
        test_case.assertTrue(
            np.allclose(ref_out[0].numpy(), out.numpy(), rtol=rtol, atol=atol)
        )
    else:
        test_case.assertTrue(len(ref_out) == len(out))
        for i in range(len(ref_out)):
            test_case.assertTrue(
                np.allclose(ref_out[i].numpy(), out[i].numpy(), rtol=rtol, atol=atol)
            )


# graph need to be a new graph, due to the limit of nn.Graph
def test_xrt_xla(test_case, graph, input, ref_out, rtol=1e-3, atol=1e-4):
    if os.getenv("ONEFLOW_TEST_FORCE_XLA") and not flow.sysconfig.with_xla():
        test_case.assertTrue(False)

    if not flow.sysconfig.with_xla():
        return

    graph.config.enable_xla_jit(True)
    if len(input) == 1:
        out = graph(input)
    else:
        out = graph(*input)

    if len(ref_out) == 1:
        test_case.assertTrue(
            np.allclose(ref_out[0].numpy(), out.numpy(), rtol=rtol, atol=atol)
        )
    else:
        test_case.assertTrue(len(ref_out) == len(out))
        for i in range(len(ref_out)):
            test_case.assertTrue(
                np.allclose(ref_out[i].numpy(), out[i].numpy(), rtol=rtol, atol=atol)
            )
