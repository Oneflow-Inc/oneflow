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
import oneflow as flow
from oneflow.test_utils.test_util import GenArgList
import oneflow.unittest
import torchvision.ops
import torch


def GetRamdomData():

    batch_sz = 30  # np.random.randint(1, 320)
    n_out_channels = 500  # np.random.randint(1, 640)
    n_in_channels = 500  # np.random.randint(1, 640)
    n_weight_grps = 1
    n_offset_grps = 1
    random_stride_h = np.random.randint(1, 5)
    random_stride_w = np.random.randint(1, 5)
    random_pad_h =np.random.randint(0, 3)
    random_pad_w =np.random.randint(0, 3)

    random_dilation_h =  np.random.randint(1, 3)
    random_dilation_w =  np.random.randint(1, 3)

    # BUG(yzm):Now use the rectangular convolution kernel is not aligned with PyTorch
    # NOTE:Added after alignment using a rectangular convolution kernel
    random_kernel_h = 5#np.random.randint(1, 11)
    random_kernel_w = 5#random_kernel_h  # np.random.randint(1, 11)

    random_in_h =  np.random.randint(1, 15)
    random_in_w =  np.random.randint(1, 15)

    stride = (random_stride_h, random_stride_w)
    pad = (random_pad_h, random_pad_w)
    dilation = (random_dilation_h, random_dilation_w)

    return (
        batch_sz,
        n_out_channels,
        n_in_channels,
        n_weight_grps,
        n_offset_grps,
        stride,
        pad,
        dilation,
        random_kernel_h,
        random_kernel_w,
        random_in_h,
        random_in_w,
    )


def GetRamdomFunArgs():
    out_w = 0
    out_h = 0
    while out_w <= 0 or out_h <= 0:
        (
            batch_sz,
            n_out_channels,
            n_in_channels,
            n_weight_grps,
            n_offset_grps,
            stride,
            pad,
            dilation,
            random_kernel_h,
            random_kernel_w,
            random_in_h,
            random_in_w,
        ) = GetRamdomData()
        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (random_kernel_h, random_kernel_w)

        in_h, in_w = (random_in_h, random_in_w)
        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

    input_np = np.random.rand(batch_sz, n_in_channels, in_h, in_w)

    offset_np = np.random.rand(batch_sz, 2 * weight_h * weight_w, out_h, out_w,)

    mask_np = np.random.rand(
        batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w
    )

    weight_np = np.random.rand(
        n_out_channels, n_in_channels // n_weight_grps, weight_h, weight_w,
    )

    bias_np = np.random.rand(n_out_channels)
    return input_np, weight_np, offset_np, mask_np, bias_np, stride, pad, dilation


def _test_deform_conv2d_forward(
    test_case,
    device,
    input_np,
    weight_np,
    offset_np,
    mask_np,
    bias_np,
    stride,
    padding,
    dilation,
):

    torch_input = torch.from_numpy(input_np).to(device)
    torch_weight = torch.from_numpy(weight_np).to(device)
    torch_offset = torch.from_numpy(offset_np).to(device)
    torch_mask = torch.from_numpy(mask_np).to(device)
    torch_bias = torch.from_numpy(bias_np).to(device)

    torch_out = torchvision.ops.deform_conv2d(
        torch_input,
        torch_offset,
        torch_weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=torch_mask,
        bias=torch_bias,
    )
    print(torch_out)
    flow_input = flow.tensor(input_np).to(device)
    flow_weight = flow.tensor(weight_np).to(device)
    flow_offset = flow.tensor(offset_np).to(device)
    flow_mask = flow.tensor(mask_np).to(device)
    flow_bias = flow.tensor(bias_np).to(device)

    flow_out = oneflow.nn.functional.deform_conv2d(
        flow_input,
        flow_offset,
        flow_weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=flow_mask,
        bias=flow_bias,
    )
    print(flow_out)
    test_case.assertTrue(
        np.allclose(flow_out.numpy(), torch_out.cpu().numpy(), rtol=1e-5, atol=1e-5,)
    )


def _test_deform_conv2d_backward(
    test_case,
    device,
    input_np,
    weight_np,
    offset_np,
    mask_np,
    bias_np,
    stride,
    padding,
    dilation,
):

    (
        input_np,
        weight_np,
        offset_np,
        mask_np,
        bias_np,
        stride,
        padding,
        dilation,
    ) = GetRamdomFunArgs()
    torch_input = torch.from_numpy(input_np).to(device).requires_grad_(True)
    torch_weight = torch.from_numpy(weight_np).to(device).requires_grad_(True)
    torch_offset = torch.from_numpy(offset_np).to(device).requires_grad_(True)
    torch_mask = torch.from_numpy(mask_np).to(device).requires_grad_(True)
    torch_bias = torch.from_numpy(bias_np).to(device).requires_grad_(True)

    torch_out = torchvision.ops.deform_conv2d(
        torch_input,
        torch_offset,
        torch_weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=torch_mask,
        bias=torch_bias,
    )
    torch_out.sum().backward()
    print("torch finish")

    flow_input = flow.tensor(input_np).to(device).requires_grad_(True)
    flow_weight = flow.tensor(weight_np).to(device).requires_grad_(True)
    flow_offset = flow.tensor(offset_np).to(device).requires_grad_(True)
    flow_mask = flow.tensor(mask_np).to(device).requires_grad_(True)
    flow_bias = flow.tensor(bias_np).to(device).requires_grad_(True)

    flow_out = oneflow.nn.functional.deform_conv2d(
        flow_input,
        flow_offset,
        flow_weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=flow_mask,
        bias=flow_bias,
    )
    flow_out.sum().backward()

    test_case.assertTrue(
        np.allclose(
            flow_input.grad.numpy(),
            torch_input.grad.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    )

    test_case.assertTrue(
        np.allclose(
            flow_weight.grad.numpy(),
            torch_weight.grad.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_offset.grad.numpy(),
            torch_offset.grad.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_mask.grad.numpy(), torch_mask.grad.cpu().numpy(), rtol=1e-5, atol=1e-5
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_bias.grad.numpy(), torch_bias.grad.cpu().numpy(), rtol=1e-5, atol=1e-5
        )
    )


def test_deform_conv2d(test_case, device):
    (
        input_np,
        weight_np,
        offset_np,
        mask_np,
        bias_np,
        stride,
        padding,
        dilation,
    ) = GetRamdomFunArgs()

    # _test_deform_conv2d_forward(
    #     test_case,
    #     device,
    #     input_np,
    #     weight_np,
    #     offset_np,
    #     mask_np,
    #     bias_np,
    #     stride,
    #     padding,
    #     dilation,
    # )
    _test_deform_conv2d_backward(
        test_case,
        device,
        input_np,
        weight_np,
        offset_np,
        mask_np,
        bias_np,
        stride,
        padding,
        dilation,
    )


@flow.unittest.skip_unless_1n1d()
class TestRoIAlign(flow.unittest.TestCase):
    def test_roi_align(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [test_deform_conv2d]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
        print("over")


if __name__ == "__main__":
    unittest.main()
