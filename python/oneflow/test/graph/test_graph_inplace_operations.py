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
import sys

from colorama import deinit
import oneflow.nn as nn
import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphInplaceOperations(flow.unittest.TestCase):
    def test_inplace_scalar_add(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self, input):
                    input += 1
                    return input

            graph = Graph()
            output = graph(flow.ones(4, 4, device=device))
            print(graph)
            print(output)

        _test("cpu")
        _test("cuda")


if __name__ == "__main__":
    unittest.main()
