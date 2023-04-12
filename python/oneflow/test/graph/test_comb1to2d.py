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


class _TestModuleDiffHierarchy(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.split(2),
        ]

        for sbp1 in sbp_1ds:

            for sbp2 in sbp_1ds:
                for sbp3 in sbp_1ds:
                    # (2, 2) -> 4
                    x = x.to_global(
                        placement=flow.placement(type="cuda", ranks=np.array(range(4))),
                        sbp=[sbp1],
                    )
                    # 4 -> (2, 2)
                    x = x.to_global(
                        placement=flow.placement(
                            type="cuda", ranks=np.array(range(4)).reshape(2, 2)
                        ),
                        sbp=[sbp2, sbp3],
                    )

        return x


class _TestModuleDiffPlacement(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.split(2),
        ]
        for sbp1 in sbp_1ds:
            for sbp2 in sbp_1ds:
                for sbp3 in sbp_1ds:
                    # (2, 2) -> 3
                    # 4 is not divisible by 3
                    x = x.to_global(
                        placement=flow.placement(type="cuda", ranks=np.array(range(3))),
                        sbp=[sbp1],
                    )
                    # 3 -> (2, 2)
                    x = x.to_global(
                        placement=flow.placement(
                            type="cuda", ranks=np.array(range(4)).reshape(2, 2)
                        ),
                        sbp=[sbp2, sbp3],
                    )

        return x


class _TestModuleDiffPlacementMiddle(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.partial_sum,
            flow.sbp.split(0),
        ]
        sbp_2ds = [
            flow.sbp.partial_sum,
            flow.sbp.split(0),
        ]
        sbp_3ds = [
            flow.sbp.split(0),
        ]

        for sbp1 in sbp_1ds:
            for sbp2 in sbp_2ds:
                for sbp3 in sbp_3ds:
                    # (2, 2) -> 3
                    # 4 is not divisible by 3
                    x = x.to_global(
                        placement=flow.placement(type="cuda", ranks=np.array(range(3))),
                        sbp=[sbp1],
                    )
                    # 3 -> (2, 2)
                    x = x.to_global(
                        placement=flow.placement(
                            type="cuda", ranks=np.array(range(4)).reshape(2, 2)
                        ),
                        sbp=[sbp2, sbp3],
                    )

        return x


class _TestModuleDiffPlacementSmall(nn.Module):
    def __init__(self, from3to4, sbp1, sbp2, sbp3) -> None:
        super().__init__()
        self.from_3to4 = from3to4
        self.sbp1 = sbp1
        self.sbp2 = sbp2
        self.sbp3 = sbp3

    def forward(self, x):
        if self.from_3to4:
            # 3 -> (2, 2)
            # 4 is not divisible by 3
            x = x.to_global(
                placement=flow.placement(type="cuda", ranks=np.array(range(3))),
                sbp=[self.sbp1],
            )
            x = x.to_global(
                placement=flow.placement(
                    type="cuda", ranks=np.array(range(4)).reshape(2, 2)
                ),
                sbp=[self.sbp2, self.sbp3],
            )
        else:
            # (2, 2) -> 3
            # 4 is not divisible by 3
            x = x.to_global(
                placement=flow.placement(
                    type="cuda", ranks=np.array(range(4)).reshape(2, 2)
                ),
                sbp=[self.sbp2, self.sbp3],
            )
            x = x.to_global(
                placement=flow.placement(type="cuda", ranks=np.array(range(3))),
                sbp=[self.sbp1],
            )

        return x


class _TestGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, x):
        x = self.model(x)
        return x


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestLazyAllSbpCombinationTesting(flow.unittest.TestCase):
    def test_lazy_boxing_2d_all_combination_diff_hierarchy(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "0"

        x = flow.ones(
            4,
            12,
            4,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
            ),
        )

        flow.boxing.nccl.enable_use_compute_stream(False)

        model_diff_hierarchy = _TestModuleDiffHierarchy()
        graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)
        y = graph_diff_hierarchy(x)

    def test_lazy_boxing_2d_all_combination_diff_placement(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "0"

        x = flow.ones(
            4,
            12,
            4,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
            ),
        )

        # use nccl logical can pass test
        flow.boxing.nccl.enable_use_compute_stream(True)
        # Got stuck.
        # flow.boxing.nccl.enable_use_compute_stream(False)

        model_diff_placement = _TestModuleDiffPlacement()
        graph_diff_placement = _TestGraph(model_diff_placement)
        z = graph_diff_placement(x)
        test_case.assertTrue(np.allclose(x.numpy(), z.numpy(), 1e-05, 1e-05))

    # This is for debug, so will not run by CI.
    def _test_lazy_boxing_2d_all_combination_diff_placement_middle(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "0"

        x = flow.ones(
            4,
            12,
            4,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
            ),
        )

        # use nccl logical can pass test
        flow.boxing.nccl.enable_use_compute_stream(True)
        # Got stuck.
        # flow.boxing.nccl.enable_use_compute_stream(False)

        model_diff_placement = _TestModuleDiffPlacementMiddle()
        graph_diff_placement = _TestGraph(model_diff_placement)
        z = graph_diff_placement(x)
        flow._oneflow_internal.eager.Sync()
        test_case.assertTrue(np.allclose(x.numpy(), z.numpy(), 1e-05, 1e-05))

    # This is for debug, so will not run by CI.
    def _test_lazy_boxing_2d_all_combination_diff_placement_small(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "0"

        x = flow.ones(
            4,
            12,
            4,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
            ),
        )

        flow.boxing.nccl.enable_use_compute_stream(False)

        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.split(2),
        ]

        for diff_p_3to4 in [True, False]:
            test_cnt = 0
            for sbp1 in sbp_1ds:
                for sbp2 in sbp_1ds:
                    for sbp3 in sbp_1ds:
                        # if flow.env.get_rank() == 0:
                        #    print("try to run ", diff_p_3to4, test_cnt, sbp1, sbp2, sbp3)
                        model_diff_placement = _TestModuleDiffPlacementSmall(
                            diff_p_3to4, sbp1, sbp2, sbp3
                        )
                        graph_diff_placement = _TestGraph(model_diff_placement)
                        z = graph_diff_placement(x)
                        test_case.assertTrue(
                            np.allclose(x.numpy(), z.numpy(), 1e-05, 1e-05)
                        )
                        # if flow.env.get_rank() == 0:
                        #    print("finish to run ", diff_p_3to4, test_cnt, sbp1, sbp2, sbp3)
                        #    print("")
                        test_cnt += 1


if __name__ == "__main__":
    unittest.main()
