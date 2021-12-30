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
import numpy as np

import oneflow as flow
import oneflow.unittest


class MyModule(flow.nn.Module):
    def __init__(self, param, placement, sbp):
        super().__init__()
        self.p = flow.nn.Parameter(
            flow.tensor(param, dtype=flow.float32, placement=placement, sbp=sbp)
        )

    def forward(self, input):
        x = flow._C.matmul(input, self.p, transpose_b=True)
        return flow.relu(x)


class MyGraph(flow.nn.Graph):
    def __init__(self, m):
        super().__init__()
        self.linear = m

    def build(self, input):
        out = self.linear(input)
        if out.is_consistent:
            out = out.to_consistent(sbp=flow.sbp.broadcast)
        return out


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class ManyGraphTestCase(oneflow.unittest.TestCase):
    @unittest.skipIf(True, "will hang")
    def test_will_hang(test_case):
        iters = 2000
        inputs = []
        params = []
        graphs = []
        placement = flow.placement("cuda", {0: [0]})
        sbp = flow.sbp.broadcast

        for _ in range(iters):
            input = np.random.rand(6, 4).astype(np.float32)
            inputs.append(input)
            param = np.random.rand(10, 4).astype(np.float32)
            params.append(param)
            m = MyModule(param, placement, sbp)
            graphs.append(MyGraph(m))

        for i, graph in enumerate(graphs):
            output = np.maximum(np.matmul(input, param.T), 0)
            input_ = flow.tensor(
                inputs[i], dtype=flow.float32, placement=placement, sbp=sbp
            )
            output_ = graph(input_)
            test_case.assertTrue(np.allclose(output_.numpy(), output))
            print(f"graph {i} finish")

    @flow.unittest.skip_unless_1n1d()
    def test_1d(test_case):
        iters = 2000
        placement = flow.placement("cuda", {0: [0]})
        sbp = flow.sbp.broadcast

        for i in range(iters):
            input = np.random.rand(6, 4).astype(np.float32)
            param = np.random.rand(10, 4).astype(np.float32)
            output = np.maximum(np.matmul(input, param.T), 0)

            m = MyModule(param, placement, sbp)
            g = MyGraph(m)

            input_ = flow.tensor(
                input, dtype=flow.float32, placement=placement, sbp=sbp
            )
            output_ = g(input_)
            test_case.assertTrue(np.allclose(output_.numpy(), output))
            print(f"graph {i} finish")
            print("output_:", output_)


if __name__ == "__main__":
    unittest.main()
