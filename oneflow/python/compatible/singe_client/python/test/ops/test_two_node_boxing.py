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
from typing import Tuple
import time


@flow.unittest.skip_unless_2n1d()
class TestTwoNodeBoxing(flow.unittest.TestCase):
    def test_two_node_boardcast(test_case):
        flow.clear_default_session()
        flow.config.enable_debug_mode(True)
        flow.config.gpu_device_num(4)
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def split_to_broadcast_job(input_blob: oft.Numpy.Placeholder((96, 96))):
            with flow.scope.placement("gpu", "0:0"):
                src = flow.identity(
                    input_blob.with_distribute(flow.distribute.split(0))
                )
            with flow.scope.placement("gpu", ["0:0", "1:0"]):
                dst = flow.identity(src.with_distribute(flow.distribute.broadcast()))
            return dst

        x = np.random.rand(96, 96).astype(np.float32)
        result = split_to_broadcast_job(x).get()
        test_case.assertTrue(np.array_equal(x, result.numpy()))


if __name__ == "__main__":
    unittest.main()
