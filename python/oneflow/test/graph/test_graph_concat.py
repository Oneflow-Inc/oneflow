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
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestConcatGraph(oneflow.unittest.TestCase):
    def test_relu_graph(test_case):
        x = flow.randn(2, 3)
        x.requires_grad = True
        y_eager = flow.cat([x], 0)

        class TestGraphOfConcat(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.layer_norm = nn.LayerNorm(x.shape[-1])
                of_sgd = flow.optim.SGD(
                    self.layer_norm.parameters(), lr=0.001, momentum=0.9,
                )
                self.add_optimizer(of_sgd)

            def build(self, x):
                res = flow.concat([x], 0)
                forward_res = res
                res = self.layer_norm(res)
                res = res.sum()
                res.backward()
                return forward_res

        concat_g = TestGraphOfConcat()
        y_lazy = concat_g(x)
        # print(f"type of lazy y: {type(y_lazy)}")
        # print(f"lazy y shape: {y_lazy.shape}, data: {y_lazy}")
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
