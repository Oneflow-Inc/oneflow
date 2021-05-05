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
from collections import OrderedDict

import os
import numpy as np
import oneflow as flow
from test_util import (
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft


def critical_section_enter(x, critical_section, group, name):
    op = (
        flow.user_op_builder(name)
        .Op("critical_section_enter")
        .Input("in", [x])
        .Output("out")
        .Attr("critical_section", critical_section)
        .Attr("group", group)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


def critical_section_leave(x, critical_section, group, name):
    op = (
        flow.user_op_builder(name)
        .Op("critical_section_leave")
        .Input("in", [x])
        .Output("out")
        .Attr("critical_section", critical_section)
        .Attr("group", group)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


def _test_critical_section(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="predict", function_config=func_config)
    def test_critical_section(
        x: oft.Numpy.Placeholder((1024, 1024), dtype=flow.float),
    ):
        with flow.scope.placement("cpu", "0:0-3"):
            x1 = x
            x2 = x
            x1 = critical_section_enter(x1, "c1", "g1", "c1-g1-enter")
            x1 = flow.math.relu(x1)
            x1 = critical_section_leave(x1, "c1", "g1", "c1-g1-leave")
            x1 = critical_section_enter(x1, "c1", "g2", "c1-g2-enter")
            x1 = flow.math.relu(x1)
            x1 = critical_section_leave(x1, "c1", "g2", "c1-g2-leave")
            x1 = critical_section_enter(x1, "c2", "g1", "c2-g1-enter")
            x1 = flow.math.relu(x1)
            x1 = critical_section_leave(x1, "c2", "g1", "c2-g1-leave")
            x1 = critical_section_enter(x1, "c2", "g2", "c2-g2-enter")
            x1 = flow.math.relu(x1)
            x1 = critical_section_leave(x1, "c2", "g2", "c2-g2-leave")

            x2 = critical_section_enter(x2, "c1", "g3", "c1-g3-enter")
            x2 = flow.math.relu(x2)
            x2 = critical_section_leave(x2, "c1", "g3", "c1-g3-leave")
            x2 = critical_section_enter(x2, "c1", "g4", "c1-g4-enter")
            x2 = flow.math.relu(x2)
            x2 = critical_section_leave(x2, "c1", "g4", "c1-g4-leave")
            x2 = critical_section_enter(x2, "c2", "g3", "c2-g3-enter")
            x2 = flow.math.relu(x2)
            x2 = critical_section_leave(x2, "c2", "g3", "c2-g3-leave")
            x2 = critical_section_enter(x2, "c2", "g4", "c2-g4-enter")
            x2 = flow.math.relu(x2)
            x2 = critical_section_leave(x2, "c2", "g4", "c2-g4-leave")
            return x1 + x2

    x = np.random.rand(1024, 1024).astype(np.float32)
    for i in range(16):
        test_critical_section(x).get()


@flow.unittest.skip_unless_1n2d()
class TestCriticalSection(flow.unittest.TestCase):
    def test(test_case):
        _test_critical_section(test_case)


if __name__ == "__main__":
    unittest.main()
