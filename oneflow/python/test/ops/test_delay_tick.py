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
import oneflow.typing as tp
import os
import unittest


@flow.unittest.skip_unless_1n1d()
class Test1dDelayTick(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_1d_no_delay(test_case):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.gpu_device_num(2)

        @flow.global_function()
        def Foo() -> tp.Numpy:
            with flow.scope.placement("gpu", device_name):
                w = flow.get_variable(
                    "w",
                    shape=(10,),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                return flow.experimental.delay_tick(w, delay_num=0)

        x = Foo()
        test_case.assertTrue(x.shape == (1,))

    def test_1d_no_delay_with_callback(test_case):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.gpu_device_num(2)

        @flow.global_function()
        def Foo() -> tp.Callback[tp.Numpy]:
            with flow.scope.placement("gpu", device_name):
                w = flow.get_variable(
                    "w",
                    shape=(10,),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                return flow.experimental.delay_tick(w, delay_num=0)

        future = Foo()
        future(lambda x: test_case.assertTrue(x.shape == (1,)))


if __name__ == "__main__":
    unittest.main()
