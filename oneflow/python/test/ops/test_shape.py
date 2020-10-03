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
import oneflow as flow
import numpy as np
import os
import random

import oneflow.typing as oft


@flow.unittest.skip_unless_1n2d()
class TestShape(flow.unittest.TestCase):
    def test_shape(test_case):
        flow.clear_default_session()
        flow.config.gpu_device_num(2)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo_job(input: oft.Numpy.Placeholder(shape=(2, 5))):
            ret = flow.identity(input)
            test_case.assertTrue(ret.shape == (1, 5))

        input_tensor = np.arange(10).reshape(2, 5).astype(np.single)
        foo_job(input_tensor)


if __name__ == "__main__":
    unittest.main()
