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
class TestSsp(flow.unittest.TestCase):
    def test_ssp(self):
        if flow.eager_execution_enabled():
            return
        device_name = "0:0"

        flow.config.enable_debug_mode(True)
        flow.config.cpu_device_num(2)

        shape = (10, 10)

        function_config = flow.FunctionConfig()
        placement = flow.scope.placement("gpu", device_name)
        function_config.ssp_stage(
            flow.stage(*[placement]*2),
            flow.stage(placement),
        )

        @flow.global_function(type="train", function_config=function_config)
        def Foo() -> tp.Numpy:
            x = flow.constant(0, dtype=flow.float, shape=shape)
            with flow.scope.placement("gpu", device_name):
                for i in range(10):
                    w = flow.get_variable(
                        "w_%s" % i,
                        shape=shape,
                        dtype=flow.float,
                        initializer=flow.constant_initializer(0),
                    )
                    x = flow.matmul(x, w)
            loss = x
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [-10.0]), momentum=0
            ).minimize(loss)
            return loss

        checkpoint = flow.train.CheckPoint()
        checkpoint.init()

        for i in range(10):
            x = Foo()


if __name__ == "__main__":
    unittest.main()
