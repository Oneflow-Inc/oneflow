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


class TestModuleDiffHierarchy(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.split(2),
            flow.sbp.split(3),
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


class TestModuleDiffPlacement(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.split(2),
            flow.sbp.split(3),
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


class TestGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, x):
        x = self.model(x)
        return x


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestLazyAllSbpCombinationTesting(flow.unittest.TestCase):
    def test_lazy_boxing_2d_all_combination(test_case):

        x = flow.ones(
            4,
            12,
            4,
            12,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
            ),
        )

        model_diff_hierarchy = TestModuleDiffHierarchy()
        graph_diff_hierarchy = TestGraph(model_diff_hierarchy)
        y = graph_diff_hierarchy(x)

        model_diff_placement = TestModuleDiffPlacement()
        graph_diff_placement = TestGraph(model_diff_placement)
        z = graph_diff_placement(x)


if __name__ == "__main__":
    unittest.main()
