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


class _TestModule(nn.Module):
    def forward(self, x):
        sbp_1ds = [
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.split(2),
        ]
        y = x

        for sbp1 in sbp_1ds:
            for sbp2 in sbp_1ds:

                for sbp3 in sbp_1ds:
                    # in this case, use intra group boxing
                    if sbp1 == sbp3:
                        continue
                    for sbp4 in sbp_1ds:
                        # (2, 2) -> (2, 2)
                        x = x.to_global(sbp=[sbp1, sbp2])
                        x = x.to_global(sbp=[sbp3, sbp4])

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
    def test_lazy_boxing_2d_all_combination(test_case):
        os.environ["ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK"] = "0"
        os.environ["ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION"] = "0"

        model = _TestModule()
        graph = _TestGraph(model)

        x = flow.ones(
            4,
            4,
            4,
            sbp=[flow.sbp.broadcast, flow.sbp.broadcast],
            placement=flow.placement(
                type="cuda", ranks=np.array(range(4)).reshape(2, 2)
            ),
        )
        y = graph(x)


if __name__ == "__main__":
    unittest.main()
