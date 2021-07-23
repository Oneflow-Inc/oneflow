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
from oneflow.compatible import single_client as flow


config = flow.function_config()


class TestTVM(unittest.TestCase):
    def test_add(self):
        a_shape = (64, 64)
        b_shape = (64, 64)
        func_config.use_tvm(True)
        func_config.use_xla_jit(False)
        func_config.use_tensorrt(False)
        @flow.global_function(function_config=func_config)
        def AddJob(a: oft.Numpy.Placeholder(a_shape), b: oft.Numpy.Placeholder(b_shape)):
            with flow.scope.placement("cpu", "0:0"):
                return a + b
        a = np.random.rand(*a_shape).astype(np.float32)
        b = np.random.rand(*b_shape).astype(np.float32)
        y = AddJob(a, b).get()
        y_np = y.numpy()
        assert np.allclose(y_np, a+b)
        flow.clear_default_session()


if __name__ == "__main__":
    unittest.main()
