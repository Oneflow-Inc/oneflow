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

if __name__ == "__main__":
    unittest.main()
