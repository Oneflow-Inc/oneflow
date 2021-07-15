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
import numpy as np
import oneflow.typing as oft
import unittest
import os


def _cpu_only_relu(x):
    op = (
        flow.user_op_builder("CpuOnlyRelu")
        .Op("cpu_only_relu_test")
        .Input("in", [x])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


def _check_cpu_only_relu_device(test_case, verbose=False):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))

    @flow.global_function(function_config=func_config)
    def cpu_only_relu_job(x_def: oft.Numpy.Placeholder(shape=(2, 5), dtype=flow.float)):
        y = _cpu_only_relu(x_def)
        if verbose:
            print("cpu_only_relu output device", y.parallel_conf.device_tag())
        test_case.assertTrue("cpu" in y.parallel_conf.device_tag())
        return y

    cpu_only_relu_job(np.random.rand(2, 5).astype(np.single)).get()


def _check_non_cpu_only_relu_device(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

    @flow.global_function(function_config=func_config)
    def relu_job(x_def: oft.Numpy.Placeholder(shape=(2, 5), dtype=flow.float)):
        with flow.scope.placement("gpu", "0:0"):
            y = flow.math.relu(x_def)

        test_case.assertTrue("gpu" in y.parallel_conf.device_tag())

        return y

    relu_job(np.random.rand(2, 5).astype(np.single)).get()


@flow.unittest.skip_unless_1n1d()
class TestCpuOnlyUserOp(flow.unittest.TestCase):
    def test_cpu_only_user_op(test_case):
        _check_cpu_only_relu_device(test_case)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_non_cpu_only_user_op(test_case):
        _check_non_cpu_only_relu_device(test_case)


if __name__ == "__main__":
    unittest.main()
