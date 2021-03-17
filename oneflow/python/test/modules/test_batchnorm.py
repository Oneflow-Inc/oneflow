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
    def test_batchnorm2d(test_case):
        input_arr = np.array(
            [
                [
                    [
                        [-0.8791, 0.2553, 0.7403, -0.2859],
                        [0.8006, -1.7701, -0.9617, 0.1705],
                        [0.2842, 1.7825, 0.3365, -0.8525],
                    ],
                    [
                        [0.7332, -0.0737, 0.7245, -0.6551],
                        [1.4461, -0.1827, 0.9737, -2.1571],
                        [0.4657, 0.7244, 0.3378, 0.1775],
                    ],
                ],
                [
                    [
                        [1.8896, 1.8686, 0.1896, 0.9817],
                        [-0.0671, 1.5569, 1.1449, 0.0086],
                        [-0.9468, -0.0124, 1.3227, -0.6567],
                    ],
                    [
                        [-0.8472, 1.3012, -1.1065, 0.9348],
                        [1.0346, 1.5703, 0.2419, -0.7048],
                        [0.6957, -0.4523, -0.8819, 1.0164],
                    ],
                ],
            ]
        )

        torch_out = np.array(
            [
                [
                    [
                        [-1.1868, -0.0328, 0.4606, -0.5833],
                        [0.5220, -2.0933, -1.2709, -0.1190],
                        [-0.0034, 1.5209, 0.0498, -1.1598],
                    ],
                    [
                        [0.5601, -0.3231, 0.5505, -0.9595],
                        [1.3404, -0.4424, 0.8233, -2.6035],
                        [0.2673, 0.5504, 0.1273, -0.0482],
                    ],
                ],
                [
                    [
                        [1.6299, 1.6085, -0.0996, 0.7062],
                        [-0.3608, 1.2914, 0.8723, -0.2837],
                        [-1.2557, -0.3051, 1.0531, -0.9606],
                    ],
                    [
                        [-1.1698, 1.1818, -1.4536, 0.7807],
                        [0.8900, 1.4763, 0.0223, -1.0139],
                        [0.5190, -0.7375, -1.2078, 0.8700],
                    ],
                ],
            ]
        )

        m = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
        m.training = True
        x = flow.Tensor(input_arr)
        y = m(x)
        print(np.allclose(y.numpy(), torch_out, atol=1e-04))


if __name__ == "__main__":
    unittest.main()
