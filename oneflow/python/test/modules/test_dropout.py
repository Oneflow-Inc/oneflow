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
        y = m(x)
        print(y.numpy())
        
        # test_dropout >> input:
        # [[-0.7797  0.2264  0.2458  0.4163]
        # [ 0.4299  0.3626 -0.4892  0.4141]
        # [-1.4115  1.2183 -0.5503  0.652 ]]
        # test_dropout >> output:
        # [[-0.      0.      0.4916  0.8326]
        # [ 0.8598  0.     -0.      0.8282]
        # [-2.823   2.4366 -0.      1.304 ]]


if __name__ == "__main__":
    unittest.main()