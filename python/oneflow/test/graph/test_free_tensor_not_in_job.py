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
import oneflow.unittest
import oneflow.nn as nn


def get_bn_graph():
    model = nn.BatchNorm1d(6)
    model.eval()
    model.to_global(flow.placement.all("cpu"), flow.sbp.broadcast)

    class Testgraph(flow.nn.Graph):
        def __init__(self, model):
            super(Testgraph, self).__init__()
            self.module = model

        def build(self, x):
            return self.module(x)

    test_graph = Testgraph(model)
    return test_graph


@flow.unittest.skip_unless_1n1d()
class TestFreeTensorNotInJob(flow.unittest.TestCase):
    def test_free_tensor_not_in_job(test_case):
        x = flow.randn(1, 6, 2).to_global(
            placement=flow.placement.all("cpu"), sbp=flow.sbp.split(0)
        )
        y = get_bn_graph()(x)
        test_case.assertEqual(y.size(), (1, 6, 2))


if __name__ == "__main__":
    unittest.main()
