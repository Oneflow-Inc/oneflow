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


def TestReshape(x, shape, name):
    return (
        flow.user_op_builder(name)
        .Op("TestReshape")
        .Input("in", [x])
        .Output("out")
        .Attr("shape", shape)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def fixed_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def ReshapeJob(x: oft.Numpy.Placeholder((10, 2))):
        return TestReshape(x, [5, 4], "xx_test_reshape")

    x = np.random.rand(10, 2).astype(np.float32)
    y = ReshapeJob(x).get().numpy()
    print(y.shape)
    test_case.assertTrue((5, 4) == y.shape)
    test_case.assertTrue(np.array_equal(x.reshape(5, 4), y))


def mirrored_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def ReshapeJob(x: oft.ListNumpy.Placeholder((10, 2))):
        return TestReshape(x, [5, 4], "xx_test_reshape")

    x = np.random.rand(8, 2).astype(np.float32)
    y = ReshapeJob([x]).get().numpy_list()[0]
    test_case.assertTrue((5, 4) == y.shape)
    reshape_x = np.concatenate((x, np.zeros((2, 2))), axis=0).reshape(5, 4)
    test_case.assertTrue(np.array_equal(reshape_x, y))


def test_fixed_TestReshape(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())
    fixed_tensor_def_test(test_case, func_config)


def test_mirrored_TestReshape(test_case):
    func_config = flow.FunctionConfig()
    mirrored_tensor_def_test(test_case, func_config)


def test_mirrored_TestReshape_1n2c(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def ReshapeJob(x: oft.ListNumpy.Placeholder((10, 2))):
        return TestReshape(x, [5, 4], "xx_test_reshape")

    x1 = np.random.rand(10, 1).astype(np.float32)
    x2 = np.random.rand(7, 2).astype(np.float32)
    y1, y2 = ReshapeJob([x1, x2]).get().numpy_list()
    test_case.assertTrue((5, 4) == y1.shape)
    test_case.assertTrue((5, 4) == y2.shape)
    reshape_x1 = np.concatenate((x1, np.zeros((10, 1)))).reshape(5, 4)
    reshape_x2 = np.concatenate((x2, np.zeros((3, 2)))).reshape(5, 4)
    test_case.assertTrue(np.array_equal(y1, reshape_x1))
    test_case.assertTrue(np.array_equal(y2, reshape_x2))
