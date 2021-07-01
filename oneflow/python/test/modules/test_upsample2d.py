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
from math import floor, ceil
from typing import Tuple

import torch
import numpy as np
import oneflow.experimental as flow
from test_util import GenArgList


def _test_upsample_and_interpolate_nearest(test_case, device, in_range, out_size_or_scale):
    print("in_range", in_range, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_size = np.sqrt(in_range[1] - in_range[0]).astype(np.int32)
    np_in = np.arange(*in_range).reshape((1, 1, in_size, in_size))
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
    )
    torch_in = torch.tensor(np_in, device=torch.device(device), dtype=torch.float32)

    m = []
    if out_size is not None:
        m.append(flow.nn.Upsample(size=out_size))
        m.append(flow.nn.interpolate(size=out_size))
        m.append(flow.nn.UpsamplingNearest2d(size=out_size))
    elif scale_factor is not None:
        m.append(flow.nn.Upsample(scale_factor=scale_factor))
        m.append(flow.nn.interpolate(scale_factor=scale_factor))
        m.append(flow.nn.UpsamplingNearest2d(scale_factor=scale_factor))
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    np_out = numpy_interpolate2d_nearest(np_in, out_size=out_size, scale_factor=scale_factor)
    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor)
    print("np_out", np_out)
    print("torch_out", torch_out)
    test_case.assertTrue(np.allclose(torch_out.cpu().numpy(), np_out, 1e-5, 1e-5))
    of_outs = []
    for it in m:
        of_outs.append(it(of_in))

    for of_out in of_outs:
        print("of_out", of_out)
        test_case.assertTrue(np.allclose(of_out.numpy(), torch_out.cpu().numpy(), 1e-5, 1e-5))


def _test_upsample_and_interpolate_bilinear(test_case, device, in_range, out_size_or_scale):
    print("in_range", in_range, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_size = np.sqrt(in_range[1] - in_range[0]).astype(np.int32)
    np_in = np.arange(*in_range, dtype=np.int32).reshape((1, 1, in_size, in_size))
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
    )
    torch_in = torch.tensor(np_in, device=torch.device(device), dtype=torch.float32)

    m = []
    if out_size is not None:
        m.append(flow.nn.Upsample(size=out_size, mode='bilinear'))
        m.append(flow.nn.interpolate(size=out_size, mode='bilinear'))
        # m.append(flow.nn.UpsamplingBilinear2d(size=out_size, align_corners=False))
    elif scale_factor is not None:
        m.append(flow.nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
        m.append(flow.nn.interpolate(scale_factor=scale_factor, mode='bilinear'))
        # m.append(flow.nn.UpsamplingBilinear2d(scale_factor=scale_factor, align_corners=False))
        # out_size = [np.floor(scale_factor * in_size).astype(np.uint8) for _ in range(2)]
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    np_out = numpy_bilinear_interpolation(np_in, out_size=out_size, scale_factor=scale_factor)
    print("np_out", np_out)
    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor, mode='bilinear')
    print("torch_out", torch_out)
    of_outs = []
    for it in m:
        of_outs.append(it(of_in))
    for of_out in of_outs:
        print("of_out", of_out)
        test_case.assertTrue(np.allclose(of_out.numpy(), torch_out.cpu().numpy(), 1e-5, 1e-5))


def _test_upsample2d_bilinear_aligncorner(test_case, device):
    input = flow.Tensor(
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
                    [1.0000, 1.3333, 1.6667, 2.0000],
                    [1.6667, 2.0000, 2.3333, 2.6667],
                    [2.3333, 2.6667, 3.0000, 3.3333],
                    [3.0000, 3.3333, 3.6667, 4.0000],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_UpsamplingNearest2d(test_case, device):
    input = flow.Tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_UpsamplingBilinear2d(test_case, device):
    input = flow.Tensor(
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
                    [1.0000, 1.3333, 1.6667, 2.0000],
                    [1.6667, 2.0000, 2.3333, 2.6667],
                    [2.3333, 2.6667, 3.0000, 3.3333],
                    [3.0000, 3.3333, 3.6667, 4.0000],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_upsample2d_4dim(test_case, device):
    input = flow.Tensor(
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
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, ],
                    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, ],
                    [4.0, 4.0, 5.0, 5.0, 6.0, 6.0, ],
                    [4.0, 4.0, 5.0, 5.0, 6.0, 6.0, ],
                    [7.0, 7.0, 8.0, 8.0, 9.0, 9.0, ],
                    [7.0, 7.0, 8.0, 8.0, 9.0, 9.0, ],
                ],
                [
                    [10.0, 10.0, 11.0, 11.0, 12.0, 12.0, ],
                    [10.0, 10.0, 11.0, 11.0, 12.0, 12.0, ],
                    [13.0, 13.0, 14.0, 14.0, 15.0, 15.0, ],
                    [13.0, 13.0, 14.0, 14.0, 15.0, 15.0, ],
                    [16.0, 16.0, 17.0, 17.0, 18.0, 18.0, ],
                    [16.0, 16.0, 17.0, 17.0, 18.0, 18.0, ],
                ],
            ],
            [
                [
                    [19.0, 19.0, 20.0, 20.0, 21.0, 21.0, ],
                    [19.0, 19.0, 20.0, 20.0, 21.0, 21.0, ],
                    [22.0, 22.0, 23.0, 23.0, 24.0, 24.0, ],
                    [22.0, 22.0, 23.0, 23.0, 24.0, 24.0, ],
                    [25.0, 25.0, 26.0, 26.0, 27.0, 27.0, ],
                    [25.0, 25.0, 26.0, 26.0, 27.0, 27.0, ],
                ],
                [
                    [28.0, 28.0, 29.0, 29.0, 30.0, 30.0, ],
                    [28.0, 28.0, 29.0, 29.0, 30.0, 30.0, ],
                    [31.0, 31.0, 32.0, 32.0, 33.0, 33.0, ],
                    [31.0, 31.0, 32.0, 32.0, 33.0, 33.0, ],
                    [34.0, 34.0, 35.0, 35.0, 36.0, 36.0, ],
                    [34.0, 34.0, 35.0, 35.0, 36.0, 36.0, ],
                ],
            ],
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_upsample2d_bilinear_4dim(test_case, device):
    input = flow.Tensor(
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
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_upsample2d_backward(test_case, device):
    input = flow.Tensor(
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
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_upsample2d_bilinear_aligncorner_backward(test_case, device):
    input = flow.Tensor(
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
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_interpolate(test_case, device):
    np_in = np.arange(1, 5).reshape((1, 1, 2, 2))
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
    )

    np_out = numpy_interpolate2d_nearest(np_in, scale_factor=2.5)
    m = flow.nn.interpolate(scale_factor=2.5, mode="nearest")
    of_out = m(of_in)
    print("of_out", of_out)
    print("np_out", np_out)
    # np_out = np.array(
    #     [
    #         [
    #             [
    #                 [1.0000, 1.1000, 1.5000, 1.9000, 2.0000],
    #                 [1.2000, 1.3000, 1.7000, 2.1000, 2.2000],
    #                 [2.0000, 2.1000, 2.5000, 2.9000, 3.0000],
    #                 [2.8000, 2.9000, 3.3000, 3.7000, 3.8000],
    #                 [3.0000, 3.1000, 3.5000, 3.9000, 4.0000]
    #             ]
    #         ]
    #     ]
    # )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_interpolate_aligncorner(test_case, device):
    input = flow.Tensor(
        np.arange(1, 5).reshape((1, 1, 2, 2)),
        device=flow.device(device),
        dtype=flow.float32,
    )
    m = flow.nn.interpolate(scale_factor=2.5, mode="bilinear", align_corners=True)
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [1.0000, 1.2500, 1.5000, 1.7500, 2.0000],
                    [1.5000, 1.7500, 2.0000, 2.2500, 2.5000],
                    [2.0000, 2.2500, 2.5000, 2.7500, 3.0000],
                    [2.5000, 2.7500, 3.0000, 3.2500, 3.5000],
                    [3.0000, 3.2500, 3.5000, 3.7500, 4.0000]
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_interpolate_backward(test_case, device):
    input = flow.Tensor(
        np.arange(1, 10).reshape((1, 1, 3, 3)),
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True,
    )

    m = flow.nn.interpolate(scale_factor=1.5)
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [
                [
                    [4., 2., 2.],
                    [2., 1., 1.],
                    [2., 1., 1.]
                ]
            ]
        ]
    )

    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def numpy_interpolate2d_nearest(img, scale_factor=None, out_size=None):
    r"""Nearest interpolate2d implemented with numpy.

        Args:
            img (Tuple[int, int, int, int]): input numpy
            scale_factor(Float): scale ratios
            out_size (Tuple[int, int]): output sizes
            img (Tuple[int, int, int, int]): output numpy

        Shape:
            - Input: :math:`(N, C{in}, H_{in}, W_{in})
            - Output: :math:`(N_{out}, C_{out},H_{out}, W_{out})` where

        .. math::
              H_{out} = \left\lfloor H_{in} \times \text{scale_factor} \right\rfloor

        .. math::
              W_{out} = \left\lfloor W_{in} \times \text{scale_factor} \right\rfloor
    """
    w = img.shape[3]
    h = img.shape[2]
    c = img.shape[1]
    n = img.shape[0]

    if out_size is not None:
        h_out = out_size[0]
        w_out = out_size[1]
        h_ratios = h_out / h
        w_ratios = w_out / w
    elif scale_factor is not None:
        assert isinstance(scale_factor, float)
        h_ratios = scale_factor
        w_ratios = scale_factor
        h_out = np.floor(scale_factor * h).astype(np.int32)
        w_out = np.floor(scale_factor * w).astype(np.int32)
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    out_img = np.zeros([n, c, h_out, w_out], dtype=np.float32)

    # mass assignment version
    y_pos = np.arange(h_out)
    x_pos = np.arange(w_out)
    y_pos = np.floor(y_pos / h_ratios).astype(np.int32)
    x_pos = np.floor(x_pos / w_ratios).astype(np.int32)
    y_pos[y_pos >= h] = h - 1
    x_pos[x_pos >= w] = w - 1

    y_pos = y_pos.reshape(y_pos.shape[0], 1)
    y_pos = np.tile(y_pos, (1, w_out))
    x_pos = np.tile(x_pos, (h_out, 1))
    assert y_pos.shape == x_pos.shape
    out_img[:, :, :, :] = img[:, :, y_pos[:, :], x_pos[:, :]]

    # navie loop version
    # for it in range(n):
    #     for ch in range(c):
    # for i in range(h_out):
    #     for j in range(w_out):
    #         org_i = min(floor(i / h_ratios), h - 1)
    #         org_j = min(floor(j / w_ratios), w - 1)
    #         out_img[:, :, i, j] = img[:, :, org_i, org_j]

    return out_img


def numpy_bilinear_interpolation(img, scale_factor=None, out_size=None):
    r"""Bilinear interpolate2d implemented with numpy.

        Args:
            img (Tuple[int, int, int, int]): input numpy
            scale_factor(Float): scale ratios
            out_size (Tuple[int, int]): output sizes
            img (Tuple[int, int, int, int]): output numpy

        Shape:
            - Input: :math:`(N, C{in}, H_{in}, W_{in})
            - Output: :math:`(N_{out}, C_{out},H_{out}, W_{out})` where

        .. math::
              H_{out} = \left\lfloor H_{in} \times \text{scale_factor} \right\rfloor

        .. math::
              W_{out} = \left\lfloor W_{in} \times \text{scale_factor} \right\rfloor
    """
    w = img.shape[3]
    h = img.shape[2]
    c = img.shape[1]
    n = img.shape[0]
    eps = 2.220446049250313e-16

    if out_size is not None:
        h_out = out_size[0]
        w_out = out_size[1]
        h_ratios = h_out / h
        w_ratios = w_out / w
    elif scale_factor is not None:
        assert isinstance(scale_factor, float)
        h_ratios = scale_factor
        w_ratios = scale_factor
        h_out = np.floor(scale_factor * h).astype(np.int32)
        w_out = np.floor(scale_factor * w).astype(np.int32)
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    out_img = np.zeros([n, c, h_out, w_out], dtype=np.float32)

    for it in range(n):
        for ch in range(c):
            for out_y in range(h_out):
                for out_x in range(w_out):
                    # in_x = round(out_x / w_ratios)  # src + 0.5 = (dst +0.5) * scale
                    # in_y = round(out_y / h_ratios)
                    in_x = (out_x + 0.5) / w_ratios - 0.5
                    in_y = (out_y + 0.5) / h_ratios - 0.5
                    if out_size == (4, 4):
                        print(in_x, in_y)
                    in_x = max(0, in_x)
                    in_y = max(0, in_y)

                    in_x_0 = floor(in_x)
                    in_y_0 = floor(in_y)
                    in_x_1 = ceil(in_x + eps)
                    in_y_1 = ceil(in_y + eps)

                    flag_x = 1
                    flag_y = 1
                    if in_x_1 > w - 1:
                        flag_x = 0
                    if in_y_1 > h - 1:
                        flag_y = 0

                    x_lambda_0 = in_x - in_x_0
                    x_lambda_1 = 1 - in_x

                    y_lambda_0 = in_y - in_y_0
                    y_lambda_1 = 1 - in_y



                    value0 = x_lambda_1 * img[it, ch, in_y_0, in_x_0] + x_lambda_0 * img[
                        it, ch, in_y_0, in_x_1]
                    value1 = y_lambda_1 * img[it, ch, in_y_1, in_x_0] + y_lambda_0 * img[
                        it, ch, in_y_1, in_x_1]
                    out_img[it, ch, out_y, out_x] = (in_y_1 - in_y) * value0 + (in_y - in_y_0) * value1
    return out_img


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestUpsample2d(flow.unittest.TestCase):
    def test_upsample2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_upsample_and_interpolate_nearest,
            _test_upsample_and_interpolate_bilinear,
            # _test_upsample2d,
            # _test_upsample2d_bilinear,
            # _test_upsample2d_bilinear_aligncorner,
            # _test_UpsamplingNearest2d,
            # _test_UpsamplingBilinear2d,
            # _test_upsample2d_4dim,
            # _test_upsample2d_bilinear_4dim,
            # _test_upsample2d_backward,
            # _test_upsample2d_bilinear_aligncorner_backward,
            # _test_interpolate,
            # _test_interpolate_aligncorner,
            # _test_interpolate_backward
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        # The squre root of the range must be an integer.
        arg_dict["in_range"] = [(1, 5), (1, 10), (5, 14)]
        arg_dict["out_size_or_scale"] = [(4, 4), (5, 5), 1.5, 0.5, 2.5]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
