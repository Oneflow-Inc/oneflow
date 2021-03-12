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
import oneflow as flow
import oneflow.typing as tp

import numpy as np
import unittest

@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_embedding(test_case):
        m = flow.nn.Embedding(6, 4)
        indices = flow.Tensor([1,3,5], dtype=flow.int32)
        print("test_embedding >> input:")
        print(m.weight.numpy(), indices.numpy())
        y = m(indices)
        print("test_embedding >> output")
        print(y.numpy())

        weight = flow.Tensor(np.random.rand(6, 4))
        m2 = flow.nn.Embedding(6, 4, _weight=weight)
        print("test_embedding with predefined weight >> input:")
        print(weight.numpy(), indices.numpy())
        y = m2(indices)
        print("test_embedding with predefined weight  >> output")
        print(y.numpy())

if __name__ == "__main__":
    unittest.main()
