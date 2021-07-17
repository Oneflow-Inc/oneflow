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
import oneflow as flow
import oneflow.typing as oft
import numpy as np
from typing import Tuple


class Add(flow.deprecated.nn.Module):
    def __init__(self):
        flow.deprecated.nn.Module.__init__(self)
        self.module_builder_ = flow.consistent_user_op_module_builder("add_n")
        self.module_builder_.InputSize("in", 2).Output("out")
        self.module_builder_.user_op_module.InitOpKernel()

    def forward(self, x, y):
        return (
            self.module_builder_.OpName("Add_%s" % self.call_seq_no)
            .Input("in", [x, y])
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )


def _make_global_func(test_case, x_shape, y_shape):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(function_config=func_config)
    def AddJob(
        x: oft.Numpy.Placeholder(shape=x_shape), y: oft.Numpy.Placeholder(shape=y_shape)
    ) -> oft.Numpy:
        with flow.scope.namespace("AddJob"):
            add_op = flow.find_or_create_module("Add", Add)
            z = add_op(x, y)
            # print(z.logical_blob_name)
            test_case.assertTrue(
                z.op_name == "AddJob-Add_{}".format(add_op.call_seq_no - 1)
            )

            v = add_op(z, x)
            # print(v.logical_blob_name)
            test_case.assertTrue(
                v.op_name == "AddJob-Add_{}".format(add_op.call_seq_no - 1)
            )

        return z

    return AddJob


@flow.unittest.skip_unless_1n1d()
class TestUserOpModule(flow.unittest.TestCase):
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

    def test_user_op_module_builder_in_namespace(test_case):
        x = np.random.rand(2, 5).astype(np.float32)
        y = np.random.rand(2, 5).astype(np.float32)

        flow.clear_default_session()
        add_func = _make_global_func(test_case, x.shape, y.shape)
        ret = add_func(x, y)
        test_case.assertTrue(np.array_equal(ret, x + y))


if __name__ == "__main__":
    unittest.main()
