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

# Enable feature to test
os.environ["ONEFLOW_DISABLE_HD_COPY_STREAM"] = "true"
import unittest
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestDisableHDCopyStreamGraph(oneflow.unittest.TestCase):
    def test_graph(test_case):
        input_arr = np.array(
            [
                [-0.00570775, 1.56608823, -1.03485088, 0.78815096],
                [-0.71527717, -1.23406712, -0.52557294, -0.460845],
                [-0.63852083, 0.86328106, 0.80663667, -0.37680572],
                [-0.08430816, -0.20784022, 0.67156639, 0.90230402],
            ],
            dtype=np.float32,
        )
        np_weight = np.ones((4, 4)).astype(np.float32)
        np_weight.fill(2.3)
        x = flow.tensor(input_arr)
        np_out = np.matmul(input_arr, np_weight)

        class MatmulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = flow.tensor(np_weight)

            def forward(self, x):
                # test H2D, D2H both
                x = x.to(flow.device("cuda"))
                w = self.weight.to(flow.device("cuda"))
                return flow.matmul(x, w).to(flow.device("cpu"))

        model = MatmulModel()
        eager_out = model(x)

        class MatmulGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model

            def build(self, x):
                return self.model(x)

        linear_g = MatmulGraph()
        of_lazy_out = linear_g(x)

        test_case.assertTrue(
            np.allclose(of_lazy_out.numpy(), eager_out.numpy(), 1e-05, 1e-05)
        )
        test_case.assertTrue(np.allclose(of_lazy_out.numpy(), np_out, 1e-05, 1e-05))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestDisableHDCopyStreamGraphWithParallel(oneflow.unittest.TestCase):
    def test_graph_with_DP(test_case):
        PLACEMENT = flow.placement("cpu", [0, 1])
        S0 = flow.sbp.split(0)
        B = flow.sbp.broadcast

        input_arr = np.array(
            [
                [
                    [1.96443248, -0.35818048, 0.8386318, -0.12787708],
                    [-0.30933945, 0.01075953, -0.85546622, 0.4462768],
                    [-0.38192172, -1.15709959, -1.33283149, -0.69705033],
                    [-0.08594144, -1.91884209, -0.56027761, 0.13399851],
                ],
                [
                    [-0.00570775, 1.56608823, -1.03485088, 0.78815096],
                    [-0.71527717, -1.23406712, -0.52557294, -0.460845],
                    [-0.63852083, 0.86328106, 0.80663667, -0.37680572],
                    [-0.08430816, -0.20784022, 0.67156639, 0.90230402],
                ],
            ],
            dtype=np.float32,
        )
        np_weight = np.ones((4, 4)).astype(np.float32)
        x = flow.tensor(input_arr, placement=PLACEMENT, sbp=S0)
        np_out = np.matmul(input_arr, np_weight)

        class MatmulModel_DP(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = flow.tensor(np_weight, placement=PLACEMENT, sbp=B)

            def forward(self, x):
                # test H2D, D2H both
                x = x.to("cuda")
                w = self.weight.to("cuda")
                return flow.matmul(x, w).to("cpu")

        model = MatmulModel_DP()

        eager_out = model(x)
        test_case.assertTrue(np.allclose(eager_out.numpy(), np_out, 1e-05, 1e-05))

        class MatmulGraph_DP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model

            def build(self, x):
                return self.model(x)

        graph = MatmulGraph_DP()
        of_lazy_out = graph(x)

        test_case.assertTrue(np.allclose(of_lazy_out.numpy(), np_out, 1e-05, 1e-05))

    def test_graph_with_MP(test_case):
        PLACEMENT = flow.placement("cpu", [0, 1])
        S1 = flow.sbp.split(1)
        B = flow.sbp.broadcast

        input_arr = np.array(
            [
                [
                    [1.96443248, -0.35818048, 0.8386318, -0.12787708],
                    [-0.30933945, 0.01075953, -0.85546622, 0.4462768],
                    [-0.38192172, -1.15709959, -1.33283149, -0.69705033],
                    [-0.08594144, -1.91884209, -0.56027761, 0.13399851],
                ],
                [
                    [-0.00570775, 1.56608823, -1.03485088, 0.78815096],
                    [-0.71527717, -1.23406712, -0.52557294, -0.460845],
                    [-0.63852083, 0.86328106, 0.80663667, -0.37680572],
                    [-0.08430816, -0.20784022, 0.67156639, 0.90230402],
                ],
            ],
            dtype=np.float32,
        )
        np_weight = np.ones((4, 4)).astype(np.float32)
        x = flow.tensor(input_arr, placement=PLACEMENT, sbp=B)
        np_out = np.matmul(input_arr, np_weight)

        class MatmulModel_MP(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = flow.tensor(np_weight, placement=PLACEMENT, sbp=S1)

            def forward(self, x):
                # test H2D, D2H both
                x = x.to("cuda")
                w = self.weight.to("cuda")
                print(x.shape)
                print(w.shape)

                return flow.matmul(x, w).to("cpu")

        model = MatmulModel_MP()

        eager_out = model(x)
        test_case.assertTrue(np.allclose(eager_out.numpy(), np_out, 1e-05, 1e-05))

        class MatmulGraph_MP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model

            def build(self, x):
                return self.model(x)

        graph = MatmulGraph_MP()
        of_lazy_out = graph(x)

        test_case.assertTrue(np.allclose(of_lazy_out.numpy(), np_out, 1e-05, 1e-05))


if __name__ == "__main__":
    unittest.main()
