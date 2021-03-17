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
import collections.abc
from itertools import repeat
import unittest
from typing import Tuple, Union

import numpy as np

import oneflow as flow
import oneflow.typing as tp


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_linear(test_case):
        linear = flow.nn.Linear(3, 8)
        input_arr = np.array(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ]
        )
        x = flow.Tensor(input_arr)
        y = linear(x)
        output_arr = np.array(
            [
                [
                    -0.01360118,
                    -0.48440266,
                    -1.7692933,
                    1.289484,
                    0.48685676,
                    1.686944,
                    0.43221387,
                    -0.55583066,
                ],
                [
                    3.0481853,
                    1.0335133,
                    0.41070116,
                    -2.0661342,
                    2.410794,
                    1.6285408,
                    3.450283,
                    -0.94717896,
                ],
                [
                    1.3638777,
                    0.09158343,
                    -0.8626478,
                    -0.24752355,
                    1.4760735,
                    1.7705703,
                    1.8076615,
                    -0.8030712,
                ],
                [
                    -1.5701474,
                    2.0831466,
                    -0.21079004,
                    1.0972698,
                    -0.25886503,
                    0.43895233,
                    -0.557955,
                    -1.1422888,
                ],
                [
                    1.4588892,
                    0.8839688,
                    -0.23292434,
                    -0.5162122,
                    1.2021716,
                    1.2807188,
                    1.9196733,
                    -0.70727587,
                ],
                [
                    -0.53847015,
                    0.65624326,
                    -1.0400524,
                    1.1920266,
                    0.07063329,
                    1.064165,
                    0.09393758,
                    -0.6831138,
                ],
                [
                    0.4123693,
                    1.0456799,
                    -0.6029235,
                    1.5609236,
                    -1.3675308,
                    0.00901711,
                    0.47493935,
                    0.70620245,
                ],
                [
                    -0.50323164,
                    1.2705305,
                    -0.41705573,
                    -0.13454032,
                    1.545763,
                    1.57799,
                    0.60342187,
                    -1.8651131,
                ],
            ]
        )
        test_case.assertEqual(y.shape, flow.Size([8, 8]))
        np.allclose(y.numpy(), output_arr, rtol=1e-04)


if __name__ == "__main__":
    unittest.main()
