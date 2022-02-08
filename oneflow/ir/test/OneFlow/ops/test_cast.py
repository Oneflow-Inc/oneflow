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
# RUN: python3 %s
import os
import unittest
import numpy as np

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"

import oneflow.compatible.single_client as flow
import oneflow.compatible.single_client.typing as oft
import oneflow.unittest
import typing

func_config = flow.FunctionConfig()


@flow.unittest.skip_unless_1n1d()
class TestCastToTosa(flow.unittest.TestCase):
    def test_idempotent(test_case):
        @flow.global_function(function_config=func_config)
        def CastJob(x: oft.Numpy.Placeholder((20, 30))) -> oft.Numpy:
            res = flow.cast(x, dtype=flow.int32)
            return res

        x = np.random.rand(20, 30).astype(np.float32) - 1
        res = CastJob(x)


if __name__ == "__main__":
    unittest.main()
