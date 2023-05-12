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

import oneflow as flow
from oneflow import nn
import os
import numpy as np

import oneflow.unittest


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphNcclLogicalFusion(flow.unittest.TestCase):
    def test_graph_nccl_fusion_1d(test_case):
        x_list = []
        local_np = np.arange(4 * 8, dtype=float).reshape(4, 8)
        P1d = flow.placement("cuda", ranks=[0, 1, 2, 3])
        B = flow.sbp.broadcast()
        S0 = flow.sbp.split(0)
        S1 = flow.sbp.split(1)
        P = flow.sbp.partial_sum()

        in_0 = (
            flow.tensor(local_np / 4.0)
            .to(flow.device("cuda"))
            .to_global(sbp=P, placement=P1d)
        )

        flow.boxing.nccl.enable_use_compute_stream(True)

        class TestNcclFusion1DGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                # fuse group 0:
                x0 = x * 0.5
                y0 = x0.to_global(sbp=B, placement=P1d)  # P->B

                x1 = x * 1.0
                y1 = x1.to_global(sbp=S0, placement=P1d)  # P->S0

                x2 = x * 2.0
                y2 = x2.to_global(sbp=S1, placement=P1d)  # P->S1

                x3 = x * 3.0
                y3 = x3.to_global(sbp=S1, placement=P1d)  # P->S1

                x4 = x * 4.0
                y4 = x4.to_global(sbp=S0, placement=P1d)  # P->S0

                # fuse group 1:
                x5 = y1 * 5.0
                y5 = x5.to_global(sbp=B, placement=P1d)  # S0->B

                x6 = y2 * (6.0 / 2.0)
                y6 = x6.to_global(sbp=B, placement=P1d)  # S1->B

                x7 = y3 * (9.0 / 3.0)
                y7 = x7.to_global(sbp=S0, placement=P1d)  # S1->S0

                x8 = y4 * (8.0 / 4.0)
                y8 = x8.to_global(sbp=S1, placement=P1d)  # S0->S1

                y = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8
                return y, y0, y1, y2, y3, y4, y5, y6, y7, y8

        graph = TestNcclFusion1DGraph()
        out, out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8 = graph(in_0)
        test_case.assertTrue(np.array_equal(out_0.numpy(), local_np * 0.5))
        test_case.assertTrue(np.array_equal(out_1.numpy(), local_np * 1.0))
        test_case.assertTrue(np.array_equal(out_2.numpy(), local_np * 2.0))
        test_case.assertTrue(np.array_equal(out_3.numpy(), local_np * 3.0))
        test_case.assertTrue(np.array_equal(out_4.numpy(), local_np * 4.0))
        test_case.assertTrue(np.array_equal(out_5.numpy(), local_np * 5.0))
        test_case.assertTrue(np.array_equal(out_6.numpy(), local_np * 6.0))
        test_case.assertTrue(np.array_equal(out_7.numpy(), local_np * 9.0))
        test_case.assertTrue(np.array_equal(out_8.numpy(), local_np * 8.0))
        flow.boxing.nccl.enable_use_compute_stream(False)

    def test_graph_nccl_fusion_2d(test_case):
        x_list = []
        local_np = np.arange(4 * 8, dtype=float).reshape(4, 8)
        P2d = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
        B = flow.sbp.broadcast()
        S0 = flow.sbp.split(0)
        S1 = flow.sbp.split(1)
        P = flow.sbp.partial_sum()

        in_BP = (
            flow.tensor(local_np / 2.0)
            .to(flow.device("cuda"))
            .to_global(sbp=(B, P), placement=P2d)
        )
        in_PB = (
            flow.tensor(local_np / 2.0)
            .to(flow.device("cuda"))
            .to_global(sbp=(P, B), placement=P2d)
        )
        in_S0P = in_BP.to_global(sbp=(S0, P), placement=P2d)
        in_PS0 = in_PB.to_global(sbp=(P, S0), placement=P2d)

        flow.boxing.nccl.enable_use_compute_stream(True)

        class TestNcclFusion2DGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x, xsd1):
                # fuse group 0:
                x0 = x * 0.5
                y0 = x0.to_global(sbp=(S0, B), placement=P2d)  # same dim0 P->B

                x1 = x * 1.0
                y1 = x1.to_global(sbp=(S0, B), placement=P2d)  # same dim0 P->B

                xss0 = x.to_global(sbp=(S0, S0), placement=P2d)
                xss1 = x.to_global(sbp=(S0, S1), placement=P2d)
                x2 = xss0 * 2.0
                y2 = x2.to_global(sbp=(S0, B), placement=P2d)  # same dim0 S0->B

                x3 = xss1 * 3.0
                y3 = x3.to_global(sbp=(S0, B), placement=P2d)  # same dim0 S1->B

                x4 = xss0 * 4.0
                y4 = x4.to_global(sbp=(S0, S1), placement=P2d)  # same dim0 S0->S1

                x5 = xss1 * 5.0
                y5 = x5.to_global(sbp=(S0, S0), placement=P2d)  # same dim0 S1->S0

                x6 = xsd1 * 6.0
                y6 = x6.to_global(sbp=(B, S0), placement=P2d)  # same dim1 P-> B

                x7 = xsd1 * 7.0
                y7 = x7.to_global(sbp=(B, S0), placement=P2d)  # same dim1 P-> B

                y = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7
                return y, y0, y1, y2, y3, y4, y5, y6, y7

        graph = TestNcclFusion2DGraph()
        out, out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7 = graph(
            in_S0P, in_PS0
        )
        test_case.assertTrue(np.array_equal(out_0.numpy(), local_np * 0.5))
        test_case.assertTrue(np.array_equal(out_1.numpy(), local_np * 1.0))
        test_case.assertTrue(np.array_equal(out_2.numpy(), local_np * 2.0))
        test_case.assertTrue(np.array_equal(out_3.numpy(), local_np * 3.0))
        test_case.assertTrue(np.array_equal(out_4.numpy(), local_np * 4.0))
        test_case.assertTrue(np.array_equal(out_5.numpy(), local_np * 5.0))
        test_case.assertTrue(np.array_equal(out_6.numpy(), local_np * 6.0))
        test_case.assertTrue(np.array_equal(out_7.numpy(), local_np * 7.0))
        flow.boxing.nccl.enable_use_compute_stream(False)


if __name__ == "__main__":
    unittest.main()
