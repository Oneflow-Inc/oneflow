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
import oneflow.compatible.single_client as flow
import oneflow.compatible.single_client.typing as oft
import typing
import os

func_config = flow.FunctionConfig()


@flow.unittest.skip_unless_1n1d()
class TestMLIROptimizations(flow.unittest.TestCase):
    def test_idempotent(test_case):
        @flow.global_function(function_config=func_config)
        def IdempotentJob(
            x: oft.Numpy.Placeholder((96, 96))
        ) -> typing.Tuple[oft.Numpy, oft.Numpy]:
            r1 = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            x = flow.math.relu(x)
            r2 = x
            return r1, r2

        x = np.random.rand(96, 96).astype(np.float32) - 1
        r1, r2 = IdempotentJob(x)
        test_case.assertTrue(np.array_equal(r1, r2))

    def test_involution(test_case):
        @flow.global_function(function_config=func_config)
        def InvolutionJob(
            x: oft.Numpy.Placeholder((96, 96))
        ) -> typing.Tuple[oft.Numpy, oft.Numpy]:
            r1 = x
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            x = flow.math.negative(x)
            r2 = x
            return r1, r2

        x = np.random.rand(96, 96).astype(np.float32) - 1
        r1, r2 = InvolutionJob(x)
        test_case.assertTrue(np.array_equal(r1, r2))


if __name__ == "__main__":
    unittest.main()
