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
import collections
import os
import unittest
from unittest import mock

import numpy as np

import oneflow as flow
import oneflow.unittest


def _run_boxing_graph(test_case, compute_around_boxing):
    local_np = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    placement = flow.placement("cuda", ranks=[0, 1])
    B = flow.sbp.broadcast()
    S0 = flow.sbp.split(0)
    S1 = flow.sbp.split(1)
    P = flow.sbp.partial_sum()

    flow.boxing.nccl.enable_use_compute_stream(True)

    class BoxingGraph(flow.nn.Graph):
        def build(self, p, s0, s1):
            if compute_around_boxing:
                p, s0, s1 = p * 1, s0 * 1, s1 * 1
            outs = (
                p.to_global(sbp=B, placement=placement),  # P->B
                p.to_global(sbp=S1, placement=placement),  # P->S1
                s0.to_global(sbp=S1, placement=placement),  # S0->S1
                s1.to_global(sbp=B, placement=placement),  # S1->B
                s1.to_global(sbp=S0, placement=placement),  # S1->S0
            )
            if compute_around_boxing:
                outs = tuple(out * 1 for out in outs)
            return outs

    x = flow.tensor(local_np, dtype=flow.bfloat16, device="cuda").to_global(
        sbp=B, placement=placement
    )
    p = flow.tensor(local_np / 2, dtype=flow.bfloat16, device="cuda").to_global(
        sbp=P, placement=placement
    )
    graph = BoxingGraph()
    outs = graph(p, x.to_global(sbp=S0), x.to_global(sbp=S1))
    for out, sbp in zip(outs, [B, S1, S1, B, S0]):
        test_case.assertEqual(out.dtype, flow.bfloat16)
        test_case.assertEqual(out.sbp, (sbp,))
        test_case.assertTrue(np.array_equal(out.float().numpy(), local_np))
    return collections.Counter(
        op.user_conf.op_type_name
        for op in graph._compiled_graph_proto.net.op
        if op.HasField("user_conf")
    )


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphNcclLogicalBfloat16(flow.unittest.TestCase):
    def test_collective_boxing_bfloat16(test_case):
        # Boxing between graph inputs and outputs goes through collective boxing.
        op_types = _run_boxing_graph(test_case, compute_around_boxing=False)
        test_case.assertFalse(
            any(t.startswith("_nccl_logical") for t in op_types), op_types
        )

    def test_nccl_logical_fusion_bfloat16(test_case):
        op_types = _run_boxing_graph(test_case, compute_around_boxing=True)
        test_case.assertIn("_nccl_logical_fusion", op_types)
        test_case.assertIn("_nccl_logical_s2s", op_types)

    def test_nccl_logical_bfloat16(test_case):
        with mock.patch.dict(os.environ, {"ENABLE_NCCL_LOGICAL_FUSION": "0"}):
            op_types = _run_boxing_graph(test_case, compute_around_boxing=True)
        test_case.assertNotIn("_nccl_logical_fusion", op_types)
        for op_type in [
            "_nccl_logical_all_reduce",
            "_nccl_logical_reduce_scatter_noncontinuous",
            "_nccl_logical_all_gather_noncontinuous",
            "_nccl_logical_s2s",
        ]:
            test_case.assertIn(op_type, op_types)


if __name__ == "__main__":
    unittest.main()
