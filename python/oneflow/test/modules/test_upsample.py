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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


def _test_upsample2d_bilinear(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
    )
    m = flow.nn.Upsample(scale_factor=2.0, mode="bilinear")
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.25, 1.75, 2.0],
                    [1.5, 1.75, 2.25, 2.5],
                    [2.5, 2.75, 3.25, 3.5],
                    [3.0, 3.25, 3.75, 4.0],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_upsample2d_bilinear_aligncorner(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
    )
    m = flow.nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.3333, 1.6667, 2.0],
                    [1.6667, 2.0, 2.3333, 2.6667],
                    [2.3333, 2.6667, 3.0, 3.3333],
                    [3.0, 3.3333, 3.6667, 4.0],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


def _test_UpsamplingNearest2d(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
    )
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_UpsamplingBilinear2d(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
    )
    m = flow.nn.UpsamplingBilinear2d(scale_factor=2.0)
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.3333, 1.6667, 2.0],
                    [1.6667, 2.0, 2.3333, 2.6667],
                    [2.3333, 2.6667, 3.0, 3.3333],
                    [3.0, 3.3333, 3.6667, 4.0],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


def _test_upsample2d_4dim(test_case, device):
    input = flow.tensor(
        np.arange(1, 37).reshape((2, 2, 3, 3)),
        device=flow.device(device),
        dtype=flow.float32,
    )
    m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                    [4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
                    [4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
                    [7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
                    [7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
                ],
                [
                    [10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
                    [10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
                    [13.0, 13.0, 14.0, 14.0, 15.0, 15.0],
                    [13.0, 13.0, 14.0, 14.0, 15.0, 15.0],
                    [16.0, 16.0, 17.0, 17.0, 18.0, 18.0],
                    [16.0, 16.0, 17.0, 17.0, 18.0, 18.0],
                ],
            ],
            [
                [
                    [19.0, 19.0, 20.0, 20.0, 21.0, 21.0],
                    [19.0, 19.0, 20.0, 20.0, 21.0, 21.0],
                    [22.0, 22.0, 23.0, 23.0, 24.0, 24.0],
                    [22.0, 22.0, 23.0, 23.0, 24.0, 24.0],
                    [25.0, 25.0, 26.0, 26.0, 27.0, 27.0],
                    [25.0, 25.0, 26.0, 26.0, 27.0, 27.0],
                ],
                [
                    [28.0, 28.0, 29.0, 29.0, 30.0, 30.0],
                    [28.0, 28.0, 29.0, 29.0, 30.0, 30.0],
                    [31.0, 31.0, 32.0, 32.0, 33.0, 33.0],
                    [31.0, 31.0, 32.0, 32.0, 33.0, 33.0],
                    [34.0, 34.0, 35.0, 35.0, 36.0, 36.0],
                    [34.0, 34.0, 35.0, 35.0, 36.0, 36.0],
                ],
            ],
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_upsample2d_bilinear_4dim(test_case, device):
    input = flow.tensor(
        np.arange(1, 37).reshape((2, 2, 3, 3)),
        device=flow.device(device),
        dtype=flow.float32,
    )
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_upsample2d_backward(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[4.0, 4.0], [4.0, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_upsample2d_bilinear_aligncorner_backward(test_case, device):
    input = flow.tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[3.999999523162842, 4.000000476837158], [3.999999761581421, 4.0]]]]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_nearest_float_scale(test_case, device):
    input = flow.tensor(
        np.arange(1, 10).reshape((1, 1, 3, 3)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.Upsample(scale_factor=1.5)
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.0, 1.0, 2.0, 3.0],
                    [1.0, 1.0, 2.0, 3.0],
                    [4.0, 4.0, 5.0, 6.0],
                    [7.0, 7.0, 8.0, 9.0],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[[[4.0, 2.0, 2.0], [2.0, 1.0, 1.0], [2.0, 1.0, 1.0]]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_interpolate_bilinear_float_scale(test_case, device):
    input = flow.tensor(
        np.arange(1, 5, dtype=np.int32).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.Upsample(scale_factor=0.5, mode="bilinear")
    of_out = m(input)
    np_out = np.array([[[[2.5]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[[[0.25, 0.25], [0.25, 0.25]]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))
    input = flow.tensor(
        np.arange(1, 10, dtype=np.int32).reshape((1, 1, 3, 3)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.Upsample(scale_factor=0.5, mode="bilinear")
    of_out = m(input)
    np_out = np.array([[[[3.0]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[[[0.25, 0.25, 0.0], [0.25, 0.25, 0.0], [0.0, 0.0, 0.0]]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))
    input = flow.tensor(
        np.arange(1, 11, dtype=np.int32).reshape((1, 1, 5, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.Upsample(size=(4, 4), mode="bilinear")
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.25, 1.5, 2.0, 2.25],
                    [3.75, 4.0, 4.5, 4.75],
                    [6.25, 6.5, 7.0, 7.25],
                    [8.75, 9.0, 9.5, 9.75],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [[[[1.75, 1.75], [1.5, 1.5], [1.5, 1.5], [1.5, 1.5], [1.75, 1.75]]]]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


def _test_upsample_bilinear_align_corners(test_case, device):
    input = flow.tensor(
        np.arange(1, 5, dtype=np.int32).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )
    m = flow.nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
    of_out = m(input)
    np_out = np.array([[[[1.0]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[[[1.0, 0.0], [0.0, 0.0]]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestUpsample2d(flow.unittest.TestCase):
    def test_upsample2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_upsample2d_bilinear,
            _test_upsample2d_bilinear_aligncorner,
            _test_UpsamplingNearest2d,
            _test_UpsamplingBilinear2d,
            _test_upsample2d_4dim,
            _test_upsample2d_bilinear_4dim,
            _test_upsample2d_backward,
            _test_upsample2d_bilinear_aligncorner_backward,
            _test_interpolate_nearest_float_scale,
            _test_interpolate_bilinear_float_scale,
            _test_upsample_bilinear_align_corners,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skip(
        "The nearest interpolate operation in pytorch has bug, https://github.com/pytorch/pytorch/issues/65200"
    )
    @autotest()
    def test_upsample2d_nearest(test_case):
        device = random_device()
        x = random_tensor().to(device)
        m = torch.nn.Upsample(scale_factor=random().to(float), mode="nearest")
        y = m(x)
        return y

    @unittest.skip(
        "The nearest interpolate operation in pytorch has bug, https://github.com/pytorch/pytorch/issues/65200"
    )
    @autotest()
    def test_upsample2d_nearest_half(test_case):
        device = random_device()
        x = random_tensor().to(device=device, dtype=torch.float16)
        m = torch.nn.Upsample(scale_factor=random().to(float), mode="nearest")
        y = m(x)
        return y

    # The forward and backward result in cpu and cuda of bilinear interpolate operation in PyTorch is different
    # in some corner cases. OneFlow has the same cpu and cuda results with PyTorch's cuda result.
    # So here we only test cuda device forward result.
    @autotest(n=10, auto_backward=False, atol=1e-8)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample2d_bilinear(test_case):
        x = random_tensor(ndim=4).to("cuda")
        x = x.permute(1, 3, 0, 2)
        m = torch.nn.Upsample(
            scale_factor=random().to(float),
            mode="bilinear",
            align_corners=random_bool(),
        )
        y = m(x)
        return y

    @autotest(atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample2d_bicubic(test_case):
        x = random_tensor(ndim=4, dim0=16, dim1=8).to("cuda")
        m = torch.nn.Upsample(
            scale_factor=random().to(float),
            mode="bicubic",
            align_corners=random_bool(),
        )
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample1d_nearest_output_size(test_case):
        x = random_tensor(ndim=3, dim0=1, dim1=2, dim2=12).to("cuda")
        m = torch.nn.Upsample(size=(13), mode="nearest")
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample2d_nearest_output_size(test_case):
        x = random_tensor(ndim=4, dim0=1, dim1=1, dim2=1, dim3=937).to("cuda")
        m = torch.nn.Upsample(size=(1, 30), mode="nearest")
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample3d_nearest_output_size(test_case):
        x = random_tensor(ndim=5, dim0=1, dim1=1, dim2=6, dim3=12, dim4=6).to("cuda")
        m = torch.nn.Upsample(size=(8, 10, 7), mode="nearest")
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample1d_linear_output_size(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim0=1, dim1=2, dim2=12).to(device)
        m = torch.nn.Upsample(size=(13), mode="linear")
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample2d_bilinear_output_size(test_case):
        x = random_tensor(ndim=4, dim0=1, dim1=1, dim2=12, dim3=21).to("cuda")
        m = torch.nn.Upsample(size=(14, 19), mode="bilinear")
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample2d_bicubic_output_size(test_case):
        x = random_tensor(ndim=4, dim0=1, dim1=2, dim2=12, dim3=21).to("cuda")
        m = torch.nn.Upsample(size=(14, 19), mode="bicubic")
        y = m(x)
        return y

    @autotest(n=5, atol=1e-5)
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_upsample3d_trilinear_output_size(test_case):
        x = random_tensor(ndim=5, dim0=1, dim1=2, dim2=1, dim3=12, dim4=17).to("cuda")
        m = torch.nn.Upsample(size=(1, 14, 23), mode="trilinear")
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()
