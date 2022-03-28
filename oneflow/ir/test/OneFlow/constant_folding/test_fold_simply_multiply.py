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
# RUN: python3 %s

import os
import unittest

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"

import numpy as np
import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn


class MultiplyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(flow.tensor([2, 2], dtype=flow.float32), False)
        self.y = nn.Parameter(flow.tensor([3, 3], dtype=flow.float32), False)

    def forward(self):
        return self.x * self.y


def _test_fold_multiply(test_case):
    model = MultiplyModel()

    eager_res = model()

    class MultiplyGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self):
            return self.model()

    graph = MultiplyGraph()
    lazy_res = graph()
    test_case.assertTrue(
        np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-5, atol=1e-5)
    )


@flow.unittest.skip_unless_1n1d()
class TestFoldMultiply(oneflow.unittest.TestCase):
    def test_fold_multiply(test_case):
        _test_fold_multiply(test_case)


if __name__ == "__main__":
    unittest.main()
