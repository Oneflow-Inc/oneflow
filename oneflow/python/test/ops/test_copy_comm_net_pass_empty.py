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
import unittest
import os


def ccrelu(x, name):
    return (
        flow.user_op_builder(name)
        .Op("ccrelu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@flow.unittest.skip_unless_2n1d()
class TestCopyCommNetPassEmpty(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_multi_node_comm_net(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        func_config.default_data_type(flow.float)
        flow.config.gpu_device_num(1)

        @flow.global_function(function_config=func_config)
        def ReluJob(x: oft.Numpy.Placeholder((10, 2))):
            with flow.scope.placement("gpu", "0:0"):
                out0 = ccrelu(x, "my_op_0_0")
            with flow.scope.placement("gpu", "1:0"):
                out1 = ccrelu(out0, "my_op_1_0")
            with flow.scope.placement("gpu", "0:0"):
                out2 = ccrelu(out1, "my_op_print")
            return out2

        index = [-2, -1, 0, 1, 2]
        data = []
        for i in index:
            data.append(np.ones((10, 2,), dtype=np.float32) * i)
        for i in range(5):
            ret = ReluJob(data[i]).get().numpy()
            print(ret)
            if index[i] > 0:
                test_case.assertTrue(
                    np.array_equal(ret, np.ones((10, 2,), dtype=np.float32) * index[i])
                )
            else:
                test_case.assertTrue(
                    np.array_equal(ret, np.zeros((10, 2,), dtype=np.float32))
                )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_multi_node_comm_net_dynamic(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))
        func_config.default_data_type(flow.float)
        flow.config.machine_num(2)
        flow.config.gpu_device_num(1)

        @flow.global_function(function_config=func_config)
        def ReluJob(x: oft.ListNumpy.Placeholder((10, 2))):
            with flow.scope.placement("gpu", "0:0"):
                out0 = flow.math.relu(x)
            with flow.scope.placement("gpu", "1:0"):
                out1 = flow.math.relu(out0)
            with flow.scope.placement("gpu", "0:0"):
                out2 = flow.math.relu(out1)
            return out2

        index = [-2, -1, 0, 1, 2]
        data = []
        for i in index:
            data.append(np.ones((5, 2,), dtype=np.float32) * i)
        for i in range(5):
            ret = ReluJob([data[i]]).get().numpy_list()[0]
            print(ret)
            if index[i] > 0:
                test_case.assertTrue(
                    np.array_equal(ret, np.ones((5, 2,), dtype=np.float32) * index[i])
                )
            else:
                test_case.assertTrue(
                    np.array_equal(ret, np.zeros((5, 2,), dtype=np.float32))
                )

    def test_multi_node_comm_net_dynamic_empty(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))
        func_config.default_data_type(flow.float)
        flow.config.machine_num(2)
        flow.config.gpu_device_num(1)

        @flow.global_function(function_config=func_config)
        def ReluJob(x: oft.ListNumpy.Placeholder((10, 2))):
            with flow.scope.placement("cpu", "0:0"):
                out0 = flow.math.relu(x)
            with flow.scope.placement("cpu", "1:0"):
                out1 = flow.math.relu(out0)
            with flow.scope.placement("cpu", "0:0"):
                out2 = flow.math.relu(out1)
            return out2

        index = [-2, -1, 0, 1, 2]
        data = []
        for i in index:
            data.append(np.ones((0, 0,), dtype=np.float32) * i)
        for i in range(5):
            ret = ReluJob([data[i]]).get().numpy_list()[0]
            print(ret)
            if index[i] > 0:
                test_case.assertTrue(
                    np.array_equal(ret, np.ones((0, 0,), dtype=np.float32) * index[i])
                )
            else:
                test_case.assertTrue(
                    np.array_equal(ret, np.zeros((0, 0,), dtype=np.float32))
                )


if __name__ == "__main__":
    unittest.main()
