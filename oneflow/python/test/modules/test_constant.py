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
import oneflow.typing as tp


flow.env.init()
flow.enable_eager_execution()

@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_ones(test_case):
        m = flow.nn.Ones()
        shape1 = (1,2,3,4)
        y1 = m(shape1)
        test_case.assertTrue(np.allclose(np.ones(shape1), y1.numpy()))


    def test_zeros(test_case):
        m = flow.nn.Zeros()
        shape = (3,2,5,1)
        y = m(shape)
        test_case.assertTrue(np.allclose(np.zeros(shape), y.numpy()))

if __name__ == "__main__":
    unittest.main()
