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

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


@flow.unittest.skip_unless_1n1d()
class TestIdentityN(flow.unittest.TestCase):
    def test_identity_n(test_case):
        @flow.global_function(type="train", function_config=func_config)
        def nvtx_range_job(x: oft.Numpy.Placeholder((4, 4, 1024, 1024))):
            x += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x = flow.math.relu(x)
            x = flow.nvtx_start(x, mark_prefix="range1")
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nvtx_end(x, mark_prefix="range1")
            x = flow.math.relu(x)
            x = flow.nvtx_start(x, mark_prefix="range2")
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.nvtx_end(x, mark_prefix="range2")
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
            ).minimize(x)
            return flow.identity(x)

        input = np.random.rand(4, 4, 1024, 1024).astype(np.float32)
        for i in range(3):
            res = nvtx_range_job(input).get()
            # test_case.assertTrue(np.array_equal(res.numpy(), input))


if __name__ == "__main__":
    unittest.main()
