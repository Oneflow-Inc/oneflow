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
import numpy as np
import os
import random
import oneflow.typing as oft
from collections import OrderedDict


def fake_flow_ones(shape):
    tensor = flow.Tensor(*shape)
    tensor.set_data_initializer(flow.ones_initializer())
    return tensor


@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mirrored_numpy4(test_case):
        np_arr = np.random.random((1000, 1000))
        while True:
            x = flow.Tensor(np_arr)
            y = x.numpy()

        func_config = flow.FunctionConfig()
        # func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def job():
            x = flow.Tensor(np.random.random((1000, 1000)))
            while True:
                y = x.numpy()

        job()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mirrored_numpy3(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def job():
            tensor = flow.Tensor(
                2, 3, data_initializer=flow.ones_initializer(flow.float32)
            )
            print(tensor.numpy())

        job()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mirrored_numpy2(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def job():
            x1 = flow.Tensor([[1, 2]])
            x2 = flow.Tensor([[1], [2]])
            op = (
                flow.builtin_op("matmul")
                .Input("a")
                .Input("b")
                .Attr("transpose_a", False)
                .Attr("transpose_b", False)
                .Attr("alpha", float(1.0))
                .Output("out")
                .Build()
            )
            y = op(x1, x2)[0]
            print(y.numpy())

        job()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mirrored_numpy(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def job():
            shape = (2, 3)
            op = (
                flow.builtin_op("constant")
                .Output("out")
                .Attr("floating_value", 1.0)
                .Attr("integer_value", 0)
                .Attr("is_floating_value", True)
                .Attr("dtype", flow.float32)
                .Attr("shape", shape)
                .Build()
            )
            tensor = op()[0]
            test_case.assertTrue(
                np.array_equal(tensor.numpy(), np.ones(shape, dtype=np.float32))
            )
            tensor.copy_(np.array([[4, 5, 6], [7, 8, 9]]))
            print(tensor.numpy())

        job()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_numpy(test_case):
        shape = (2, 3)
        test_case.assertTrue(
            np.array_equal(
                fake_flow_ones(shape).numpy(), np.ones(shape, dtype=np.float32)
            )
        )

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_init(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape)

        x.fill_(5)
        test_case.assertTrue(np.array_equal(x.numpy(), 5 * np.ones(x.shape)))

        flow.nn.init.ones_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np.ones(x.shape)))

        z = flow.Tensor(5, 5)
        flow.nn.init.kaiming_normal_(z, a=0.1, mode="fan_out", nonlinearity="relu")
        flow.nn.init.kaiming_uniform_(z)
        flow.nn.init.xavier_normal_(z)
        flow.nn.init.xavier_uniform_(z)

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_creating_consistent_tensor(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=flow.placement("gpu", ["0:0"], None))
        x.set_placement(flow.placement("cpu", ["0:0"], None))
        x.set_is_consistent(True)
        test_case.assertTrue(not x.is_cuda)
        x.determine()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_user_defined_data(test_case):
        list_data = [5, 5]
        tuple_data = (5, 5)
        numpy_data = np.array((5, 5))
        x = flow.Tensor(list_data)
        y = flow.Tensor(tuple_data)
        z = flow.Tensor(numpy_data)

        test_case.assertTrue(np.array_equal(x.numpy(), 5 * np.ones(x.shape)))
        test_case.assertTrue(np.array_equal(y.numpy(), 5 * np.ones(y.shape)))
        test_case.assertTrue(np.array_equal(z.numpy(), 5 * np.ones(z.shape)))


if __name__ == "__main__":
    unittest.main()
