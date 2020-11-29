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
import numpy as np
import oneflow as flow
import oneflow.typing as oft


def _test_two_job_optimizer_placement_optimization(test_case):
    flow.config.gpu_device_num(2)
    flow.config.enable_debug_mode(True)
    eval_config = flow.FunctionConfig()
    eval_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(eval_config)
    def Bar():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        return w

    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.optimizer_placement_optimization_mode("non_distributed")

    @flow.global_function(type="train", function_config=func_config)
    def Foo(x: oft.Numpy.Placeholder((2, 10))):
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [5]), momentum=0
        ).minimize(x + w)

    Foo(np.ones((2, 10), dtype=np.float32))


def _test_optimizer_placement_optimization_var_as_loss(test_case):
    flow.config.gpu_device_num(2)
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.optimizer_placement_optimization_mode("non_distributed")

    @flow.global_function(type="train", function_config=func_config)
    def Foo():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(100))
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [5]), momentum=0
        ).minimize(w)

    Foo()


@flow.unittest.skip_unless_1n2d()
class TestOptimizerPlacementOptimization(flow.unittest.TestCase):
    def test_optimizer_placement_optimization(test_case):
        flow.config.gpu_device_num(2)
        flow.config.enable_debug_mode(True)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        func_config.optimizer_placement_optimization_mode("non_distributed")

        @flow.global_function(type="train", function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((2, 10))):
            w = flow.get_variable(
                "w", (10,), initializer=flow.constant_initializer(100)
            )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [5]), momentum=0
            ).minimize(x + w)

        Foo(np.ones((2, 10), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
