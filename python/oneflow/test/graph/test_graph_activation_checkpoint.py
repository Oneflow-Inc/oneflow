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
import re
import os
import unittest

import numpy as np

import oneflow
import oneflow as flow
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.framework.scope_util as scope_util
import oneflow.unittest
from oneflow.nn.graph import GraphModule


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphActivationCheckpoint(flow.unittest.TestCase):
    def test_activation_checkpoint(test_case):
        loss_fn = flow.nn.MSELoss(reduction="sum")
        model = flow.nn.Sequential(flow.nn.Linear(3, 4), flow.nn.Linear(4, 4))
        model1 = flow.nn.Sequential(flow.nn.Linear(4, 1), flow.nn.Flatten(0, 1))

        class SubModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, x):
                scope = scope_util.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                test_case.assertEqual(ck_bool, True)
                out = self.model(x)
                return out

        class SubModule1(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model1

            def forward(self, x):
                scope = scope_util.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                test_case.assertEqual(ck_bool, True)
                out = self.model(x)
                return out

        optimizer = flow.optim.SGD(model.parameters(), lr=1e-6)

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = SubModule0()
                self.model1 = SubModule1()
                self.loss_fn = loss_fn
                # Add an optimizer
                self.add_optimizer(optimizer)
                self.model.to(GraphModule).activation_checkpointing = True
                self.model1.to(GraphModule).activation_checkpointing = True

            def build(self, x, y):
                y_pred = self.model(x)
                y_pred = self.model1(y_pred)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                return loss

        linear_graph = LinearTrainGraph()
        x = flow.randn(10, 3)
        y = flow.randn(10)
        linear_graph._compile(x, y)

        graph_proto = linear_graph._full_graph_proto
        for op in graph_proto.net.op:
            # Check flatten gradient operator take checkpoiting as input
            if re.search("flatten.*grad", op.name, re.I) is not None:
                find_check_point = False
                for value in op.user_conf.input.values():
                    if (
                        re.search("Sys-Checkpointing-Fake-Fw-Op", str(value), re.I)
                        is not None
                    ):
                        find_check_point = True
                        print(value)
                test_case.assertTrue(find_check_point)
            # Check having insert identity op and first fake op of a segment has indentity grad as it's ctrl in op
            if (
                re.search(
                    "Sys-Checkpointing-Fake-Fw-Op_model.model.0-matmul*", op.name, re.I,
                )
                is not None
            ):
                find_ctrl = False
                for name in op.ctrl_in_op_name:
                    if re.search("identity", str(name), re.I) is not None:
                        find_ctrl = True
                        print(name)
                test_case.assertTrue(find_ctrl)


if __name__ == "__main__":
    unittest.main()
