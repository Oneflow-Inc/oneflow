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
import torchvision.ops
import torch

import oneflow as flow
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import random_tensor
from oneflow.test_utils.test_util import GenArgList
import oneflow.unittest


def GetRandomData(max_batch_sz):
    batch_sz = max_batch_sz
    n_weight_grps = np.random.randint(1, 2)
    n_offset_grps = np.random.randint(1, 2)
    n_out_channels = n_offset_grps * np.random.randint(1, 15)
    n_in_channels = n_offset_grps * np.random.randint(1, 15)

    random_stride_h = np.random.randint(1, 5)
    random_stride_w = np.random.randint(1, 5)
    random_pad_h = np.random.randint(0, 3)
    random_pad_w = np.random.randint(0, 3)
    random_dilation_h = np.random.randint(1, 3)
    random_dilation_w = np.random.randint(1, 3)
    random_in_h = np.random.randint(5, 30)
    random_in_w = np.random.randint(5, 30)

    # BUG(yzm): Now use the rectangular convolution kernel is not aligned with PyTorch
    # NOTE: Modify the following program after alignment using a rectangular convolution kernel
    random_kernel_h = np.random.randint(1, 11)
    random_kernel_w = random_kernel_h
    # random_kernel_w=np.random.randint(1, 11)

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


def GetFunArgs(device, max_batch_size):
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
        ) = GetRandomData(max_batch_size)
        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (random_kernel_h, random_kernel_w)
        in_h, in_w = (random_in_h, random_in_w)
        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

    input_dims = [batch_sz, n_in_channels, in_h, in_w]
    offset_dims = [batch_sz, 2 * n_offset_grps * weight_h * weight_w, out_h, out_w]
    mask_dims = [batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w]
    weight_dims = [n_out_channels, n_in_channels // n_weight_grps, weight_h, weight_w]

    input = random_tensor(4, *input_dims).to(device)
    offset = random_tensor(4, *offset_dims).to(device)
    mask = random_tensor(4, *mask_dims).to(device)
    weight = random_tensor(4, *weight_dims).to(device)
    bias_dims = [n_out_channels]
    bias = random_tensor(1, *bias_dims).to(device)
    return input, weight, offset, mask, bias, stride, pad, dilation


def _test_deform_conv2d_forward(
    test_case, input, weight, offset, mask, bias, stride, padding, dilation,
):
    torch_input = input.pytorch
    torch_weight = weight.pytorch
    torch_offset = offset.pytorch
    torch_mask = mask.pytorch
    torch_bias = bias.pytorch

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

    flow_input = input.oneflow
    flow_weight = weight.oneflow
    flow_offset = offset.oneflow
    flow_mask = mask.oneflow
    flow_bias = bias.oneflow

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
    test_case.assertTrue(
        np.allclose(
            flow_out.numpy(), torch_out.detach().cpu().numpy(), rtol=1e-2, atol=1e-2
        )
    )


def _test_deform_conv2d_backward(
    test_case, input, weight, offset, mask, bias, stride, padding, dilation
):
    torch_input = input.pytorch.detach().requires_grad_()
    torch_weight = weight.pytorch.detach().requires_grad_()
    torch_offset = offset.pytorch.detach().requires_grad_()
    torch_mask = mask.pytorch.detach().requires_grad_()
    torch_bias = bias.pytorch.detach().requires_grad_()

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

    flow_input = input.oneflow.detach().requires_grad_()
    flow_weight = weight.oneflow.detach().requires_grad_()
    flow_offset = offset.oneflow.detach().requires_grad_()
    flow_mask = mask.oneflow.detach().requires_grad_()
    flow_bias = bias.oneflow.detach().requires_grad_()

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
            rtol=1e-2,
            atol=1e-2,
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_weight.grad.numpy(),
            torch_weight.grad.cpu().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_offset.grad.numpy(),
            torch_offset.grad.cpu().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_mask.grad.numpy(), torch_mask.grad.cpu().numpy(), rtol=1e-2, atol=1e-2
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_bias.grad.numpy(), torch_bias.grad.cpu().numpy(), rtol=1e-5, atol=1e-5
        )
    )


def _test_forward_and_backward(test_case, device):
    max_batch_size = 40
    for batch_size in range(1, max_batch_size):
        input, weight, offset, mask, bias, stride, padding, dilation = GetFunArgs(
            device, batch_size
        )
        _test_deform_conv2d_forward(
            test_case, input, weight, offset, mask, bias, stride, padding, dilation
        )
        _test_deform_conv2d_backward(
            test_case, input, weight, offset, mask, bias, stride, padding, dilation
        )


@flow.unittest.skip_unless_1n1d()
class TestDeformConv2d(flow.unittest.TestCase):
    def test_deform_conv2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_forward_and_backward]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
