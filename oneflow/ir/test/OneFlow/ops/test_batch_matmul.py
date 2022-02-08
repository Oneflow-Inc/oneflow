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


@unittest.skip("doens't work for now")
@flow.unittest.skip_unless_1n1d()
class TestMatMulCPUToTosa(flow.unittest.TestCase):
    def test_idempotent(test_case):
        @flow.global_function(function_config=func_config)
        def BatchMatMulJob(
            x: oft.Numpy.Placeholder((1, 20, 30)), y: oft.Numpy.Placeholder((1, 30, 20))
        ) -> oft.Numpy:
            with flow.scope.placement("cpu", "0:0-0"):
                res = flow.matmul(x, y)
                return res

        x = np.random.rand(1, 20, 30).astype(np.float32) - 1
        y = np.random.rand(1, 30, 20).astype(np.float32) - 1
        res = BatchMatMulJob(x, y)
        print(res.shape)
        test_case.assertTrue(
            np.allclose(res.flatten(), np.matmul(x, y).flatten(), rtol=1e-4)
        )


@unittest.skip("doens't work for now")
@flow.unittest.skip_unless_1n1d()
class TestMatMulGPUToTosa(flow.unittest.TestCase):
    def test_idempotent(test_case):
        @flow.global_function(function_config=func_config)
        def BatchMatMulJob(
            x: oft.Numpy.Placeholder((1, 20, 30)), y: oft.Numpy.Placeholder((1, 30, 20))
        ) -> oft.Numpy:
            with flow.scope.placement("gpu", "0:0-0"):
                res = flow.matmul(x, y)
                return res

        x = np.random.rand(1, 20, 30).astype(np.float32) - 1
        y = np.random.rand(1, 30, 20).astype(np.float32) - 1
        res = BatchMatMulJob(x, y)
        print(res.shape)
        test_case.assertTrue(
            np.allclose(res.flatten(), np.matmul(x, y).flatten(), rtol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
