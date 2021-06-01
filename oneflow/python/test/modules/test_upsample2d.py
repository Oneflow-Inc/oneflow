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
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestUpsample2d(flow.unittest.TestCase):
    def test_upsample2d(test_case):
        input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0, 1.0, 2.0, 2.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0],
                        [3.0, 3.0, 4.0, 4.0],
                    ]
                ]
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_upsample2d_bilinear(test_case):
        input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.Upsample(scale_factor=2.0, mode="bilinear")
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0000, 1.2500, 1.7500, 2.0000],
                        [1.5000, 1.7500, 2.2500, 2.5000],
                        [2.5000, 2.7500, 3.2500, 3.5000],
                        [3.0000, 3.2500, 3.7500, 4.0000],
                    ]
                ]
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_upsample2d_bilinear_aligncorner(test_case):
        input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0000, 1.3333, 1.6667, 2.0000],
                        [1.6667, 2.0000, 2.3333, 2.6667],
                        [2.3333, 2.6667, 3.0000, 3.3333],
                        [3.0000, 3.3333, 3.6667, 4.0000],
                    ]
                ]
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-3, 1e-3))

    def test_UpsamplingNearest2d(test_case):
        input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.UpsamplingNearest2d(scale_factor=2.0)
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0, 1.0, 2.0, 2.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0],
                        [3.0, 3.0, 4.0, 4.0],
                    ]
                ]
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_UpsamplingBilinear2d(test_case):
        input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.UpsamplingBilinear2d(scale_factor=2.0)
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0000, 1.3333, 1.6667, 2.0000],
                        [1.6667, 2.0000, 2.3333, 2.6667],
                        [2.3333, 2.6667, 3.0000, 3.3333],
                        [3.0000, 3.3333, 3.6667, 4.0000],
                    ]
                ]
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-3, 1e-3))

    def test_upsample2d_4dim(test_case):
        input = flow.Tensor(np.arange(1, 37).reshape((2, 2, 3, 3)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0,],
                        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0,],
                        [4.0, 4.0, 5.0, 5.0, 6.0, 6.0,],
                        [4.0, 4.0, 5.0, 5.0, 6.0, 6.0,],
                        [7.0, 7.0, 8.0, 8.0, 9.0, 9.0,],
                        [7.0, 7.0, 8.0, 8.0, 9.0, 9.0,],
                    ],
                    [
                        [10.0, 10.0, 11.0, 11.0, 12.0, 12.0,],
                        [10.0, 10.0, 11.0, 11.0, 12.0, 12.0,],
                        [13.0, 13.0, 14.0, 14.0, 15.0, 15.0,],
                        [13.0, 13.0, 14.0, 14.0, 15.0, 15.0,],
                        [16.0, 16.0, 17.0, 17.0, 18.0, 18.0,],
                        [16.0, 16.0, 17.0, 17.0, 18.0, 18.0,],
                    ],
                ],
                [
                    [
                        [19.0, 19.0, 20.0, 20.0, 21.0, 21.0,],
                        [19.0, 19.0, 20.0, 20.0, 21.0, 21.0,],
                        [22.0, 22.0, 23.0, 23.0, 24.0, 24.0,],
                        [22.0, 22.0, 23.0, 23.0, 24.0, 24.0,],
                        [25.0, 25.0, 26.0, 26.0, 27.0, 27.0,],
                        [25.0, 25.0, 26.0, 26.0, 27.0, 27.0,],
                    ],
                    [
                        [28.0, 28.0, 29.0, 29.0, 30.0, 30.0,],
                        [28.0, 28.0, 29.0, 29.0, 30.0, 30.0,],
                        [31.0, 31.0, 32.0, 32.0, 33.0, 33.0,],
                        [31.0, 31.0, 32.0, 32.0, 33.0, 33.0,],
                        [34.0, 34.0, 35.0, 35.0, 36.0, 36.0,],
                        [34.0, 34.0, 35.0, 35.0, 36.0, 36.0,],
                    ],
                ],
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_upsample2d_bilinear_4dim(test_case):
        input = flow.Tensor(np.arange(1, 37).reshape((2, 2, 3, 3)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.Upsample(scale_factor=2.0, mode="bilinear")
        of_out = m(input)
        np_out = np.array(
            [
                [
                    [
                        [1.0, 1.25, 1.75, 2.25, 2.75, 3.0],
                        [1.75, 2.0, 2.5, 3.0, 3.5, 3.75],
                        [3.25, 3.5, 4.0, 4.5, 5.0, 5.25],
                        [4.75, 5.0, 5.5, 6.0, 6.5, 6.75],
                        [6.25, 6.5, 7.0, 7.5, 8.0, 8.25],
                        [7.0, 7.25, 7.75, 8.25, 8.75, 9.0],
                    ],
                    [
                        [10.0, 10.25, 10.75, 11.25, 11.75, 12.0],
                        [10.75, 11.0, 11.5, 12.0, 12.5, 12.75],
                        [12.25, 12.5, 13.0, 13.5, 14.0, 14.25],
                        [13.75, 14.0, 14.5, 15.0, 15.5, 15.75],
                        [15.25, 15.5, 16.0, 16.5, 17.0, 17.25],
                        [16.0, 16.25, 16.75, 17.25, 17.75, 18.0],
                    ],
                ],
                [
                    [
                        [19.0, 19.25, 19.75, 20.25, 20.75, 21.0],
                        [19.75, 20.0, 20.5, 21.0, 21.5, 21.75],
                        [21.25, 21.5, 22.0, 22.5, 23.0, 23.25],
                        [22.75, 23.0, 23.5, 24.0, 24.5, 24.75],
                        [24.25, 24.5, 25.0, 25.5, 26.0, 26.25],
                        [25.0, 25.25, 25.75, 26.25, 26.75, 27.0],
                    ],
                    [
                        [28.0, 28.25, 28.75, 29.25, 29.75, 30.0],
                        [28.75, 29.0, 29.5, 30.0, 30.5, 30.75],
                        [30.25, 30.5, 31.0, 31.5, 32.0, 32.25],
                        [31.75, 32.0, 32.5, 33.0, 33.5, 33.75],
                        [33.25, 33.5, 34.0, 34.5, 35.0, 35.25],
                        [34.0, 34.25, 34.75, 35.25, 35.75, 36.0],
                    ],
                ],
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


if __name__ == "__main__":
    unittest.main()
