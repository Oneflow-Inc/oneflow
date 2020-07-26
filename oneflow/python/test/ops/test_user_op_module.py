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
import oneflow as flow
import oneflow.typing as oft
import numpy as np
from typing import Tuple


def test_Add(test_case):
    @flow.global_function()
    def AddJob(xs: Tuple[(oft.Numpy.Placeholder((5, 2)),) * 2]):
        adder = flow.find_or_create_module("Add", Add)
        x = adder(*xs)
        y = adder(*xs)
        return adder(x, y)

    inputs = tuple(np.random.rand(5, 2).astype(np.float32) for i in range(2))
    r = AddJob(inputs).get().numpy()
    test_case.assertTrue(np.allclose(r, sum(inputs) * 2))
    r = AddJob(inputs).get().numpy()
    test_case.assertTrue(np.allclose(r, sum(inputs) * 2))


def test_find_or_create_module_reuse(test_case):
    @flow.global_function()
    def AddJob(xs: Tuple[(oft.Numpy.Placeholder((5, 2)),) * 2]):
        adder = flow.find_or_create_module("Add", Add, reuse=True)
        adder = flow.find_or_create_module("Add", Add, reuse=True)
        x = adder(*xs)
        return adder(x, x)

    inputs = tuple(np.random.rand(5, 2).astype(np.float32) for i in range(2))
    r = AddJob(inputs).get().numpy()


class Add(flow.nn.Module):
    def __init__(self):
        flow.nn.Module.__init__(self)
        self.module_builder_ = flow.consistent_user_op_module_builder("add_n")
        self.module_builder_.InputSize("in", 2).Output("out")
        self.module_builder_.user_op_module.InitOpKernel()

    def forward(self, x, y):
        unique_id = flow.current_scope().auto_increment_id()
        return (
            self.module_builder_.OpName("add_n_%s" % unique_id)
            .Input("in", [x, y])
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
