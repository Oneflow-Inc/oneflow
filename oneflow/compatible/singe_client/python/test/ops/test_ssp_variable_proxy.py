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
import oneflow.typing as tp
import os
import unittest


@flow.unittest.skip_unless_1n1d()
class Test1dSspVariableProxy(flow.unittest.TestCase):
    def test_1d_ring_buffer_Wm_assign_Wc_plus_1(test_case):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.cpu_device_num(2)

        buffer_size = 4

        @flow.global_function()
        def Foo() -> tp.Numpy:
            with flow.scope.placement("cpu", device_name):
                w = flow.get_variable(
                    "w",
                    shape=(10,),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                ones = flow.constant_like(w, value=1.0, dtype=flow.float)
                ref, value = flow.experimental.ssp_variable_proxy(
                    w, buffer_size=buffer_size
                )
                # do no use `w` again because it's delegated by `ref` and `value`
                # W_mutable = W_constant + 1
                flow.assign(ref, value + ones)
                return value

        zeros = np.zeros((10,)).astype(np.float32)
        ones = np.ones((10,)).astype(np.float32)

        # the first four results are always initialized with zeros
        for i in range(buffer_size):
            x = Foo()
            test_case.assertTrue(np.allclose(x, zeros))

        # the next for results are ones, because the formula is W_mutable = W_constant + 1
        for i in range(buffer_size):
            x = Foo()
            test_case.assertTrue(np.allclose(x, ones))

        # the next for results are twos, because the formula is W_mutable = W_constant + 1
        for i in range(buffer_size):
            x = Foo()
            test_case.assertTrue(np.allclose(x, ones + ones))

    def test_1d_ring_buffer_Wm_assign_Wm_plus_1(test_case):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.cpu_device_num(2)

        buffer_size = 4

        @flow.global_function()
        def Foo() -> tp.Numpy:
            with flow.scope.placement("cpu", device_name):
                w = flow.get_variable(
                    "w",
                    shape=(10,),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                ones = flow.constant_like(w, value=1.0, dtype=flow.float)
                ref, value = flow.experimental.ssp_variable_proxy(
                    w, buffer_size=buffer_size
                )
                # do no use `w` again because it's delegated by `ref` and `value`
                # W_mutable = W_mutable + 1
                flow.assign(ref, ref + ones)
                return value

        zeros = np.zeros((10,)).astype(np.float32)
        ones = np.ones((10,)).astype(np.float32)

        # the first four results are always initialized with zeros
        for i in range(buffer_size):
            x = Foo()
            test_case.assertTrue(np.allclose(x, zeros))
        # ones, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones))

        # twos, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones + ones))

        # threes, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones + ones + ones))

        # fours, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones + ones + ones + ones))

    def test_add_ssp_variable_proxy(test_case):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.enable_debug_mode(True)
        flow.config.cpu_device_num(2)

        buffer_size = 4

        function_config = flow.FunctionConfig()
        function_config.enable_ssp(True)

        @flow.global_function(type="train", function_config=function_config)
        def Foo() -> tp.Numpy:
            with flow.scope.placement(
                "cpu", device_name
            ), flow.experimental.scope.config(
                ssp_num_stages=buffer_size, ssp_stage_id=0
            ):
                w = flow.get_variable(
                    "w",
                    shape=(10,),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                loss = w + flow.constant_like(w, value=0.0, dtype=flow.float)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [-10.0]), momentum=0
                ).minimize(loss)
                return loss

        zeros = np.zeros((10,)).astype(np.float32)
        ones = np.ones((10,)).astype(np.float32)

        # the first four results are always initialized with zeros
        for i in range(buffer_size):
            x = Foo()
            test_case.assertTrue(np.allclose(x, zeros))
        # ones, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones))

        # twos, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones + ones))

        # threes, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones + ones + ones))

        # fours, because the formula is W_mutable = W_mutable + 1
        x = Foo()
        test_case.assertTrue(np.allclose(x, ones + ones + ones + ones))


if __name__ == "__main__":
    unittest.main()
