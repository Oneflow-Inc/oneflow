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
import numpy as np
import unittest

# used to observe operator optimization and execution order manually
# import os
# os.environ["ONEFLOW_DEBUG_MODE"] = "1"
# os.environ["GLOG_v"] = "3"
# os.environ["ENABLE_LOGICAL_CHAIN"] = "true"

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

# NOTE: nn.functional.depend() behaves differently in the two modes
# in EAGER mode, the OP has no effect. That is, the first paramerter
# and output are the same tensor (like "y=x" in python), while the
# second paramerter will be ignore.


def _build_graph_and_test(TestModel, in_data, test_case):

    model = TestModel()
    y_eager = model(in_data)

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    graph = TestGraph()
    # used to observe operator optimization and execution order manually
    # graph.debug(3)
    y_lazy = graph(in_data)
    test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestDependGraph(oneflow.unittest.TestCase):
    def test_depend_graph_case0(test_case):
        class TestModel_0(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x * 2" be executed before "self.linear(x)" in graph mode
                # base use case
                x1 = x * 2
                x = nn.functional.depend(x, x1)
                x2 = self.linear(x)
                return x2

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_0, x, test_case)

    def test_depend_graph_case1(test_case):
        class TestModel_1(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x * 2" and "x + 2" be executed before "self.linear(x)" in graph mode
                # test multiple continuous nn.functional.depend() in a logical chain
                x1 = x * 2
                x2 = x + 2
                x = nn.functional.depend(x, x1)
                x = nn.functional.depend(x, x2)
                x3 = self.linear(x)
                return x3

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_1, x, test_case)

    def test_depend_graph_case2(test_case):
        class TestModel_2(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x * 2" and "x + 2" be executed before "self.linear(x)" in graph mode
                # some users may code like this
                x1 = x * 2
                x2 = x + 2
                x2 = nn.functional.depend(x2, x1)
                x = nn.functional.depend(x, x2)
                x3 = self.linear(x)
                return x3

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_2, x, test_case)

    def test_depend_graph_case3(test_case):
        class TestModel_3(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x * 2", "x + 2" and "x -2" be executed before "self.linear(x)" in graph mode
                # a combination of above cases
                x1 = x * 2
                x2 = x + 2
                x3 = x - 2
                x = nn.functional.depend(x, x1)
                x2 = nn.functional.depend(x2, x3)
                x = nn.functional.depend(x, x2)
                x3 = self.linear(x)
                return x3

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_3, x, test_case)

    def test_depend_graph_case4(test_case):
        class TestModel_4(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # the depend OP do nothing and it should be pruned from graph correctly
                x1 = x * 2
                x2 = nn.functional.depend(x, x1)
                x3 = self.linear(x)
                return x3

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_4, x, test_case)

    def test_depend_graph_case5(test_case):
        class TestModel_5(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(128, 128)
                self.linear1 = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x * 2" be executed before "self.linear0(x)" and
                # "self.linear1(x)" in graph mode
                # to test the case that depend OP connect to more than one OPs
                x1 = x * 2
                x = nn.functional.depend(x, x1)
                x2 = self.linear0(x)
                x3 = self.linear1(x)
                return x2 + x3

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_5, x, test_case)

    def test_depend_graph_case6(test_case):
        class TestModel_6(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x - 2" be executed before "self.linear(x)" in graph mode
                # to test the case that the OP connects to Depend OP also connects to other OPs
                x1 = x * 2
                x2 = x1 - 2
                x3 = nn.functional.depend(x2, x1)
                x4 = self.linear(x3)
                x5 = x2 + x4
                return x5

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_6, x, test_case)

    def test_depend_graph_case7(test_case):
        class TestModel_7(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # to ensure "mp_values * 2" be executed before "max_pool1d" in graph mode
                # to test the case that OPs have mutiple outputs connect to depend OP
                x1 = x + 2
                mp_values, mp_indices = nn.functional.max_pool1d(
                    x, kernel_size=2, return_indices=True
                )
                mp_values = nn.functional.depend(mp_values, x1)
                mp_values = mp_values * 2
                return mp_values + mp_indices.to(flow.float32)

        x = flow.randn([1, 2, 3], dtype=flow.float32)
        _build_graph_and_test(TestModel_7, x, test_case)

    def test_depend_graph_case8(test_case):
        class TestModel_1(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # to ensure "x * 2" and "x + 2" be executed before "self.linear(x)" in graph mode
                # to test the case that inputting mutiple depend tensors at a time
                x1 = x * 2
                x2 = x + 2
                x = nn.functional.depend(x, [x1, x2])
                x3 = self.linear(x)
                return x3

        x = flow.randn([1, 128], dtype=flow.float32)
        _build_graph_and_test(TestModel_1, x, test_case)


if __name__ == "__main__":
    unittest.main()
