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
func_config.default_logical_view(flow.scope.consistent_view())


@flow.unittest.skip_unless_1n1d()
class TestKeepHeaderOnlyCpu(flow.unittest.TestCase):
    def test_keep_header_only_cpu(test_case):
        @flow.global_function(function_config=func_config)
        def job(x: oft.Numpy.Placeholder((2, 3, 4), dtype=flow.float)):
            with flow.scope.placement("cpu", "0:0"):
                x = flow.identity(x)
                return flow.math.reduced_shape_elem_cnt(x)

        test_case.assertTrue(
            job(np.zeros((2, 3, 4), np.float32)).get().item() == 2 * 3 * 4
        )


if __name__ == "__main__":
    unittest.main()
