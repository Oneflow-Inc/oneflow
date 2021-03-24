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
    def test_layernorm(test_case):
        input_arr = np.array(
                [[[[ 0.1706, -0.8850],
                [-0.4255, -1.3691]],

                [[-0.6181,  0.2705],
                [ 0.0179, -0.9482]]],


                [[[ 0.0342,  0.8918],
                [ 0.8133, -0.2475]],

                [[-0.1996,  1.0143],
                [-0.7829,  0.7150]]]], dtype = np.float32
        )

        torch_out = np.array(
                [[[[ 1.1684, -0.7469],
                [ 0.0868, -1.6253]],

                [[-0.2626,  1.3497],
                [ 0.8914, -0.8616]]],


                [[[-0.3955,  0.9854],
                [ 0.8590, -0.8491]],

                [[-0.7720,  1.1826],
                [-1.7112,  0.7007]]]], dtype = np.float32
        )

        m = flow.nn.LayerNorm([2,2,2])
        x = flow.Tensor(input_arr)
        y = m(x)
        print(np.allclose(y.numpy(), torch_out, atol=1e-04))


if __name__ == "__main__":
    unittest.main()
