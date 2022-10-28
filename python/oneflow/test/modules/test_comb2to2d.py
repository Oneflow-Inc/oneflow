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


flow.boxing.nccl.enable_use_compute_stream(False)


class _TestModuleDiffHierarchy(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
        ]

        for sbp1 in sbp_1ds:
            for sbp2 in sbp_1ds:

                for sbp3 in sbp_1ds:
                    for sbp4 in sbp_1ds:
                        # (3, 2) -> (2, 3)
                        x = x.to_global(
                            placement=flow.placement(
                                type="cuda", ranks=np.array(range(6)).reshape(2, 3)
                            ),
                            sbp=[sbp1, sbp2],
                        )
                        # (2, 3) -> (3, 2)
                        x = x.to_global(
                            placement=flow.placement(
                                type="cuda", ranks=np.array(range(6)).reshape(3, 2)
                            ),
                            sbp=[sbp3, sbp4],
                        )

        return x


class _TestModuleDiffPlacement(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
        ]

        for sbp1 in sbp_1ds:
            for sbp2 in sbp_1ds:

                for sbp3 in sbp_1ds:
                    for sbp4 in sbp_1ds:
                        # (3, 2) -> (2, 2)
                        x = x.to_global(
                            placement=flow.placement(
                                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
                            ),
                            sbp=[sbp1, sbp2],
                        )
                        # (2, 2) -> (3, 2)
                        x = x.to_global(
                            placement=flow.placement(
                                type="cuda", ranks=np.array(range(6)).reshape(3, 2)
                            ),
                            sbp=[sbp3, sbp4],
                        )

        return x


class _TestGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, x):
        x = self.model(x)
        return x


@flow.unittest.skip_unless_2n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestLazyAllSbpCombinationTesting(flow.unittest.TestCase):
    def test_lazy_boxing_2d_all_combination(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "0"

        x = flow.ones(
            12,
            12,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(6)).reshape(3, 2)
            ),
        )

        model_diff_hierarchy = _TestModuleDiffHierarchy()
        graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)
        y = graph_diff_hierarchy(x)

        model_diff_placement = _TestModuleDiffPlacement()
        graph_diff_placement = _TestGraph(model_diff_placement)
        z = graph_diff_placement(x)


if __name__ == "__main__":
    unittest.main()
