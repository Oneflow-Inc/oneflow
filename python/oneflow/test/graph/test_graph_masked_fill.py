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
import random

import oneflow as flow
from oneflow import nn
import oneflow.unittest
from test_util import generate_graph


@flow.unittest.skip_unless_1n1d()
class TestMaskedFillGraph(flow.unittest.TestCase):
    def test_masked_fill_graph(test_case):
        k = random.randint(1, 10)
        model = nn.Sequential(nn.Linear(k, k))
        optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        class MaskedFillGraph(flow.nn.Graph):
            def __init__(self,):
                super().__init__()
                self.model = model
                self.loss_fn = loss_fn
                self.add_optimizer(optimizer)

            def build(self, input, mask):
                output = self.model(input)
                output = flow.masked_fill(output, mask > 0.5, 0.5)
                loss = self.loss_fn(output, input)
                loss.backward()
                return loss

        input = flow.randn(k, k).requires_grad_()
        mask = flow.randn(k, k)
        model = MaskedFillGraph()
        return model(input, mask)

    def test_masked_fill_by_generate_graph(test_case):
        k = random.randint(1, 10)
        input = flow.randn(k, k)
        mask = flow.randn(k, k)

        masked_fill_fn = lambda: flow.masked_fill(input, mask > 0.5, 0.5)
        y_eager = masked_fill_fn()
        masked_fill_graph = generate_graph(masked_fill_fn)
        y_lazy = masked_fill_graph()
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
