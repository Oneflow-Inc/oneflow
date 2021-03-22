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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_sigmoid(test_case):
        m = flow.nn.Flatten()
        x = flow.Tensor(32, 2, 5, 5, data_initializer=flow.random_normal_initializer())
        y = m(x)
        print(y.shape[1] == 50)
        print(np.allclose(y.numpy().flatten(), x.numpy().flatten()))

        y2 = x.flatten(2)
        print(y2.shape[2] == 25)
        print(np.allclose(y2.numpy().flatten(), x.numpy().flatten()))

if __name__ == "__main__":
    unittest.main()
