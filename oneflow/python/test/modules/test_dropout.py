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
    

    def test_dropout(test_case):
        m = flow.nn.Dropout(p=0.5)
        input_arr = np.array(
            [[-0.7797,  0.2264,  0.2458,  0.4163],
            [ 0.4299,  0.3626, -0.4892,  0.4141],
            [-1.4115,  1.2183, -0.5503,  0.6520]]
        )
        x = flow.Tensor(np.array(
            [[-0.7797,  0.2264,  0.2458,  0.4163],
            [ 0.4299,  0.3626, -0.4892,  0.4141],
            [-1.4115,  1.2183, -0.5503,  0.6520]]
        ))
        print("test_dropout >> input:\n", x.numpy())
        y = m(x)
        print("test_dropout >> output:\n", y.numpy())


if __name__ == "__main__":
    unittest.main()