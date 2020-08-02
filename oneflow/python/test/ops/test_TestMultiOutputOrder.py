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
import oneflow as flow
import oneflow.typing as oft


def TestMultiOutputOrder(x, name):
    return (
        flow.user_op_builder(name)
        .Op("TestMultiOutputOrder")
        .Input("in", [x])
        .Output("out1")
        .Output("out2")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )


def GenerateTest(test_case, shape):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(function_config=func_config)
    def TestMultiOutputOrderJob(x: oft.Numpy.Placeholder(shape)):
        return TestMultiOutputOrder(x, "my_2_output_op")

    x = np.random.rand(*shape).astype(np.float32)
    # print("x", x)
    out1, out2 = TestMultiOutputOrderJob(x).get()
    out1_ndarray = out1.numpy()
    out2_ndarray = out2.numpy()
    # print("out1", out1_ndarray)
    # print("out2", out2_ndarray)
    out2_shape = list(shape)
    out2_shape[-1] = out2_shape[-1] * 2
    out2_shape = tuple(out2_shape)
    test_case.assertTrue(shape == out1_ndarray.shape)
    test_case.assertTrue(out2_shape == out2_ndarray.shape)
    test_case.assertTrue(np.allclose(x, out1_ndarray))
    test_case.assertTrue(
        np.allclose(np.zeros(out2_shape, dtype=np.float32), out2_ndarray)
    )


def test_TestMultiOutputOrder_example_1(test_case):
    GenerateTest(test_case, (7,))


def test_TestMultiOutputOrder_example_2(test_case):
    GenerateTest(test_case, (2, 5,))


def test_TestMultiOutputOrder_example_3(test_case):
    GenerateTest(test_case, (3, 3, 2,))
