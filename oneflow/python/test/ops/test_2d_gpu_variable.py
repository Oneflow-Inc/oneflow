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
import os
import unittest


@unittest.skipIf(flow.unittest.env.node_size() != 1, "only runs when node_size is 1")
@unittest.skipIf(flow.unittest.env.device_num() != 2, "only runs when device_num is 2")
class Test2dGpuVariable(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_2d_gpu_variable(test_case):
        flow.enable_eager_execution()
        flow.config.gpu_device_num(2)
        device_name = "0:0-1"

        @flow.global_function(type="train", function_config=flow.FunctionConfig())
        def Foo():
            with flow.scope.placement("gpu", device_name):
                w = flow.get_variable(
                    "w",
                    shape=(10,),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                print(w.numpy(0))
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.1]), momentum=0
            ).minimize(w)

        Foo()
        Foo()


if __name__ == "__main__":
    unittest.main()
