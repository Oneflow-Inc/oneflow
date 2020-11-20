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

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


@flow.unittest.skip_unless_1n1d()
class TestIdentityN(flow.unittest.TestCase):
    def test_identity_n(test_case):
        @flow.global_function(function_config=func_config)
        def identity_n_job(xs: Tuple[(oft.Numpy.Placeholder((5, 2)),) * 3]):
            return flow.identity_n(xs)

        inputs = tuple(np.random.rand(5, 2).astype(np.float32) for i in range(3))
        res = identity_n_job(inputs).get()
        for i in range(3):
            test_case.assertTrue(np.array_equal(res[i].numpy(), inputs[i]))


if __name__ == "__main__":
    unittest.main()
