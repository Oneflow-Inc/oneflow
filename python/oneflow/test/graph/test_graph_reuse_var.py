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
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphResueVar(flow.unittest.TestCase):
    def test_graph_reuse_var(test_case):
        rank = flow.env.get_rank()
        P = flow.placement("cuda", ranks=[0, 1])
        B = flow.sbp.broadcast

        class ReuseVarModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(2, 2)
                self.linear2 = flow.nn.Linear(2, 2)
                # Reuse parameter
                self.linear2.weight = self.linear1.weight

            def forward(self, x):
                # Allow user to call parameter outside it's module.
                self.linear1.weight
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        reuse_var_m = ReuseVarModule()
        reuse_var_m.to_global(placement=P, sbp=B)
        of_sgd = flow.optim.SGD(reuse_var_m.parameters(), lr=0.001, momentum=0.9)

        class ReuseVarGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.reuse_var_m = reuse_var_m
                self.add_optimizer(of_sgd)

            def build(self, x):
                x = self.reuse_var_m(x)
                loss = x.sum()
                loss.backward()
                return loss

        x = flow.randint(0, 1, (2, 2), placement=P, sbp=B, dtype=flow.float32)
        reuse_var_g = ReuseVarGraph()
        loss = reuse_var_g(x)

        # check lazy tensor builder
        block = reuse_var_g.reuse_var_m
        test_case.assertEqual(
            block.linear1.weight.lazy_origin_builder().name,
            "reuse_var_m.linear1.weight",
        )
        test_case.assertEqual(
            block.linear1.weight.lazy_origin_builder().name,
            block.linear2.weight.lazy_origin_builder().name,
        )

        # check optimizer's variable list
        var_list = [
            "reuse_var_m.linear1.weight",
            "reuse_var_m.linear1.bias",
            "reuse_var_m.linear2.bias",
        ]
        var_list_in_conf = reuse_var_g._graph_proto.job_conf.train_conf.optimizer_conf[
            0
        ].variable_op_names
        test_case.assertEqual(len(var_list_in_conf), 3)
        for idx in range(3):
            test_case.assertEqual(var_list[idx], var_list_in_conf[idx])
            if rank == 0:
                print(var_list_in_conf[idx])


if __name__ == "__main__":
    unittest.main()
