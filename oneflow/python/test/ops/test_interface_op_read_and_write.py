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
import os
import unittest

import numpy as np

import oneflow as flow
import oneflow.typing as tp


@flow.unittest.skip_unless_1n2d()
class TestInterfaceOpReadAndWrite(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test(test_case):
        flow.config.gpu_device_num(2)

        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return

        @flow.global_function()
        def add() -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0-1"):
                x = flow.get_variable(
                    name="x",
                    shape=(2, 3),
                    initializer=flow.random_uniform_initializer(),
                )
                y = flow.get_variable(
                    name="y",
                    shape=(2, 3),
                    initializer=flow.random_uniform_initializer(),
                )
                return flow.math.add_n([x, y])

        # NOTE(chengcheng): Should retain for session init before set_interface_blob_value
        flow.train.CheckPoint().init()

        x_value = np.random.random((2, 3)).astype(np.float32)
        y_value = np.random.random((2, 3)).astype(np.float32)
        flow.experimental.set_interface_blob_value("x", x_value)
        flow.experimental.set_interface_blob_value("y", y_value)
        test_case.assertTrue(
            np.array_equal(x_value, flow.experimental.get_interface_blob_value("x"))
        )
        test_case.assertTrue(
            np.array_equal(y_value, flow.experimental.get_interface_blob_value("y"))
        )
        test_case.assertTrue(np.array_equal(add(), x_value + y_value))


if __name__ == "__main__":
    unittest.main()
