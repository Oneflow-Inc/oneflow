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

import numpy as np

# To enable MultiClient
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "12139"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

import oneflow
import oneflow.experimental as flow
import oneflow.python.framework.graph_build_util as graph_build_util


@flow.unittest.skip_unless_1n1d()
class TestGraph(flow.unittest.TestCase):
    def test_forward_graph(test_case):
        class CustomModule0(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = flow.nn.Conv2d(1, 1, 5)
                self.relu = flow.nn.ReLU()
                self.register_buffer(
                    "dummy_buff", flow.Tensor(1, 4),
                )
                self.register_parameter(
                    "dummy_para", flow.nn.Parameter(flow.Tensor(1, 4)),
                )

            def forward(self, x):
                self.dummy_para
                self.dummy_buff
                # x = self.conv1(x)
                # x = self.relu(x)
                return x

        m = CustomModule0()

        class CustomGraph0(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m

            def build(self, x):
                out = self.m(x)
                return out

        g = CustomGraph0()
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        print(repr(g))
        z = g._compile(x)
        print("type(z): ", type(z))
        print("graph proto: ", g._graph_proto)


if __name__ == "__main__":
    unittest.main()
