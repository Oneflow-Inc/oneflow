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
import re
import unittest

import numpy as np

import oneflow
import oneflow as flow
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.unittest


class TestGraphActivationCheckpoint(flow.unittest.TestCase):
    def test_activation_checkpoint(test_case):
        loss_fn = flow.nn.MSELoss(reduction="sum")
        model = flow.nn.Sequential(flow.nn.Linear(3, 1), flow.nn.Flatten(0, 1))
        optimizer = flow.optim.SGD(model.parameters(), lr=1e-6)

        class SubModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, x):
                scope = oneflow.current_scope()
                scope_proto = graph_build_util.scope_to_proto(scope)
                ck_bool = scope_proto.attr_name2attr_value["checkpointing"].at_bool
                test_case.assertEqual(ck_bool, True)
                out = self.model(x)
                return out

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = SubModule0()
                self.loss_fn = loss_fn
                # Add an optimizer
                self.add_optimizer(optimizer)
                self.model.config.activation_checkpointing = True

            def build(self, x, y):
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                return loss

        linear_graph = LinearTrainGraph()
        x = flow.randn(10, 3)
        y = flow.randn(10)
        linear_graph._compile(x, y)

        graph_proto = linear_graph._graph_proto
        for op in graph_proto.net.op:
            # Check flatten gradient operator take checkpoiting as input
            if re.search("flatten.*grad", op.name, re.I) is not None:
                find_check_point = False
                for value in op.user_conf.input.values():
                    if re.search("checkpointing", str(value), re.I) is not None:
                        find_check_point = True
                test_case.assertTrue(find_check_point)


if __name__ == "__main__":
    unittest.main()
