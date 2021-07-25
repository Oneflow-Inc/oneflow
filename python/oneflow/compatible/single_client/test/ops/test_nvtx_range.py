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

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


@flow.unittest.skip_unless_1n1d()
class TestProfilerNvtxRange(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_profiler_nvtx_range(test_case):
        @flow.global_function(type="train", function_config=func_config)
        def nvtx_range_job(x: oft.Numpy.Placeholder((4, 4, 1024, 1024))):
            x += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x = flow.math.relu(x)
            x = flow.profiler.nvtx_start(x, mark_prefix="softmax")
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.nn.softmax(x)
            x = flow.profiler.nvtx_end(x, mark_prefix="softmax")
            x = flow.math.relu(x)
            x = flow.profiler.nvtx_start(x, mark_prefix="gelu")
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.math.gelu(x)
            x = flow.profiler.nvtx_end(x, mark_prefix="gelu")
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
            ).minimize(x)
            return flow.identity(x)

        input = np.random.rand(4, 4, 1024, 1024).astype(np.float32)
        for i in range(3):
            res = nvtx_range_job(input).get()


if __name__ == "__main__":
    unittest.main()
