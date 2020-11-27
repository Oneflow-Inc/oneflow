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
class Test1dBufferOp(flow.unittest.TestCase):
    def test_add_ssp_variable_proxy(test_case):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.enable_debug_mode(True)
        flow.config.cpu_device_num(2)

        buffer_size = 4

        function_config = flow.FunctionConfig()
        function_config.enable_ssp(True)
        function_config.enable_stage_buffer(True)

        shape = (10, 10)

        @flow.global_function(type="train", function_config=function_config)
        def Foo() -> tp.Numpy:
            with flow.scope.placement(
                "cpu", device_name
            ), flow.experimental.scope.config(num_stages=buffer_size, stage_id=0):
                w = flow.get_variable(
                    "w",
                    shape=shape,
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                x = w + flow.constant_like(w, value=0.0, dtype=flow.float)
                loss = flow.matmul(x, x) + flow.matmul(w, w)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [-10.0]), momentum=0
                ).minimize(loss)
                return loss

        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        zeros = np.zeros(shape).astype(np.float32)
        ones = np.ones(shape).astype(np.float32)

        for i in range(buffer_size):
            x = Foo()


if __name__ == "__main__":
    unittest.main()
