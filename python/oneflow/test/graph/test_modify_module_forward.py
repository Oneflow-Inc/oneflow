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
import argparse
import numpy as np
import os
import time
import unittest
from types import MethodType

import oneflow as flow
import oneflow.unittest
from oneflow import nn


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestModifyForwardOfModule(oneflow.unittest.TestCase):
    def test_modify_forward(test_case):
        def forward2(self, x):
            return x + 1

        class Model1(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        class ForwardModifiedGraph(nn.Graph):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.model.eval()

            def build(self, x):
                return self.model(x)

        test_model = Model1()
        test_model.forward = MethodType(forward2, test_model)
        eval_graph_model1 = ForwardModifiedGraph(model=test_model)

        input_tensor = flow.tensor([0.0], requires_grad=True)

        eager_out = test_model(input_tensor)
        graph_out = eval_graph_model1(input_tensor)
        test_case.assertTrue(np.array_equal(graph_out.numpy(), eager_out.numpy()))


if __name__ == "__main__":
    unittest.main()
