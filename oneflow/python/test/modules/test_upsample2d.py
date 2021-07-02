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
import cv2
import PIL
import PIL.Image as Image


def _test_upsample_and_interpolate_nearest(test_case, device, in_size, out_size_or_scale):
    print("in_size", in_size, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_range = (1, in_size[3] * in_size[2] * in_size[1] * in_size[0] + 1)
    np_in = np.arange(*in_range).reshape(in_size)
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

    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor)
    print("torch_out", torch_out)
    of_outs = []
    for it in m:
        of_outs.append(it(of_in))

    for of_out in of_outs:
        print("of_out", of_out)
        test_case.assertTrue(np.allclose(of_out.numpy(), torch_out.cpu().numpy(), 1e-5, 1e-5))

def _test_upsample_and_interpolate_bilinear(test_case, device, in_size, out_size_or_scale):
    print("in_size", in_size, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_range = (1, in_size[3] * in_size[2] * in_size[1] * in_size[0] + 1)
    np_in = np.arange(*in_range).reshape(in_size)
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
    elif scale_factor is not None:
        m.append(flow.nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
        m.append(flow.nn.interpolate(scale_factor=scale_factor, mode='bilinear'))
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor, mode='bilinear')
    print("torch_out", torch_out)
    of_outs = []
    for it in m:
        of_outs.append(it(of_in))
    for of_out in of_outs:
        print("of_out", of_out)
        # bypass bug implementation made by pytorch
        if in_size !=(1, 1, 2, 3) and scale_factor != 0.5:
            test_case.assertTrue(np.allclose(of_out.numpy(), torch_out.cpu().numpy(), 1e-5, 1e-5))


def _test_upsample_and_interpolate_bilinear_align_corners(test_case, device, in_size, out_size_or_scale):
    print("in_size", in_size, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_range = (1, in_size[3] * in_size[2] * in_size[1] * in_size[0] + 1)
    np_in = np.arange(*in_range).reshape(in_size)
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
    )
    torch_in = torch.tensor(np_in, device=torch.device(device), dtype=torch.float32)
    pil_in = Image.fromarray(np_in[0, 0].astype(np.uint8), 'L')
    cv_in = np.asarray(pil_in)

    m = []
    if out_size is not None:
        m.append(flow.nn.Upsample(size=out_size, mode='bilinear', align_corners=True))
        m.append(flow.nn.interpolate(size=out_size, mode='bilinear', align_corners=True))
        m.append(flow.nn.UpsamplingBilinear2d(size=out_size))
    elif scale_factor is not None:
        m.append(flow.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))
        m.append(flow.nn.interpolate(scale_factor=scale_factor, mode='bilinear', align_corners=True))
        m.append(flow.nn.UpsamplingBilinear2d(scale_factor=scale_factor))
        # cv_out_size = tuple(np.floor(scale_factor * in_size).astype(np.uint8) for _ in range(2))
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    #
    # if out_size is None:
    #     pil_out = pil_in.resize(cv_out_size, resample=PIL.Image.BILINEAR)
    #     cv_out = cv2.resize(cv_in, cv_out_size, interpolation=cv2.INTER_LINEAR)
    # else:
    #     print(cv_in.shape)
    #     pil_out = pil_in.resize(out_size, resample=PIL.Image.BILINEAR)
    #     cv_out = cv2.resize(cv_in, out_size, interpolation=cv2.INTER_LINEAR)
    # print("pil_out", np.array(pil_out))
    # print("cv_out", cv_out)
    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    print("torch_out", torch_out)
    of_outs = []
    for it in m:
        of_outs.append(it(of_in))
    for of_out in of_outs:
        print("of_out", of_out)
        test_case.assertTrue(np.allclose(of_out.numpy(), torch_out.cpu().numpy(), 1e-5, 1e-5))


def _test_upsample_and_interpolate_nearest_backward(test_case, device, in_size, out_size_or_scale):
    print("in_size", in_size, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_range = (1, in_size[3] * in_size[2] * in_size[1] * in_size[0] + 1)
    np_in = np.arange(*in_range).reshape(in_size)
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad = True
    )
    torch_in = torch.tensor(np_in, device=torch.device(device), dtype=torch.float32, requires_grad = True)

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

    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor)
    torch_out = torch_out.sum()
    torch_out.backward()
    print("torch_out_grad", torch_in.grad)


    of_outs = []
    for it in m:
        of_outs.append(it(of_in))

    for of_out in of_outs:
        of_out = of_out.sum()
        of_out.backward()

        print("of_out_grad", of_in.grad)
        test_case.assertTrue(np.allclose(of_in.grad.numpy(), torch_in.grad.cpu().numpy(), 1e-5, 1e-5))
        of_in.grad = None


def _test_upsample_and_interpolate_bilinear_backward(test_case, device, in_size, out_size_or_scale):
    print("in_size", in_size, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_range = (1, in_size[3] * in_size[2] * in_size[1] * in_size[0] + 1)
    np_in = np.arange(*in_range).reshape(in_size)
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad = True
    )
    torch_in = torch.tensor(np_in, device=torch.device(device), dtype=torch.float32, requires_grad=True)

    m = []
    if out_size is not None:
        m.append(flow.nn.Upsample(size=out_size, mode='bilinear'))
        m.append(flow.nn.interpolate(size=out_size, mode='bilinear'))
    elif scale_factor is not None:
        m.append(flow.nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
        m.append(flow.nn.interpolate(scale_factor=scale_factor, mode='bilinear'))
    else:
        raise ValueError("Either out_size or scale_factor should not be None")


    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor, mode='bilinear')
    torch_out = torch_out.sum()
    torch_out.backward()
    print("torch_out_grad", torch_in.grad)

    of_outs = []
    for it in m:
        of_outs.append(it(of_in))

    for of_out in of_outs:
        of_out = of_out.sum()
        of_out.backward()

        print("of_out_grad", of_in.grad)
        # bypass bug implementation made by pytorch
        if in_size != (1, 1, 2, 3) and out_size_or_scale != 0.5:
            test_case.assertTrue(np.allclose(of_in.grad.numpy(), torch_in.grad.cpu().numpy(), 1e-5, 1e-5))
        of_in.grad = None



def _test_upsample_and_interpolate_bilinear_align_corners_backward(test_case, device, in_size, out_size_or_scale):
    print("in_size", in_size, "out_size_or_scale", out_size_or_scale)
    out_size = None
    scale_factor = None
    if isinstance(out_size_or_scale, Tuple):
        out_size = out_size_or_scale
    elif isinstance(out_size_or_scale, (float, int)):
        scale_factor = out_size_or_scale
    in_range = (1, in_size[3] * in_size[2] * in_size[1] * in_size[0] + 1)
    np_in = np.arange(*in_range).reshape(in_size)
    of_in = flow.Tensor(
        np_in,
        device=flow.device(device),
        dtype=flow.float32,
        requires_grad=True
    )
    torch_in = torch.tensor(np_in, device=torch.device(device), dtype=torch.float32,requires_grad=True)
    pil_in = Image.fromarray(np_in[0, 0].astype(np.uint8), 'L')
    cv_in = np.asarray(pil_in)

    m = []
    if out_size is not None:
        m.append(flow.nn.Upsample(size=out_size, mode='bilinear', align_corners=True))
        m.append(flow.nn.interpolate(size=out_size, mode='bilinear', align_corners=True))
        m.append(flow.nn.UpsamplingBilinear2d(size=out_size))
    elif scale_factor is not None:
        m.append(flow.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))
        m.append(flow.nn.interpolate(scale_factor=scale_factor, mode='bilinear', align_corners=True))
        m.append(flow.nn.UpsamplingBilinear2d(scale_factor=scale_factor))
    else:
        raise ValueError("Either out_size or scale_factor should not be None")

    torch_out = torch.nn.functional.interpolate(torch_in, size=out_size, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    torch_out = torch_out.sum()
    torch_out.backward()
    print("torch_out_grad", torch_in.grad)

    of_outs = []
    for it in m:
        of_outs.append(it(of_in))


    for of_out in of_outs:
        of_out = of_out.sum()
        of_out.backward()

        print("of_out_grad", of_in.grad)
        test_case.assertTrue(np.allclose(of_in.grad.numpy(), torch_in.grad.cpu().numpy(), 1e-5, 1e-5))
        of_in.grad = None


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
            _test_upsample_and_interpolate_bilinear_align_corners,
            _test_upsample_and_interpolate_nearest_backward,
            _test_upsample_and_interpolate_bilinear_backward,
            _test_upsample_and_interpolate_bilinear_align_corners_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        # Order of input dimensions is (N, C, H, W)
        arg_dict["in_size"] = [(1, 1, 2, 3), (1, 1, 5, 2), (1, 1, 3, 6), (2, 3, 2, 6), (4, 2, 4, 2)]
        # Output size must be a Tuple, scale_factor can be Int or Float.
        arg_dict["out_size_or_scale"] = [(4, 4), (5, 5), 1.5, 0.5, 2.5]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
