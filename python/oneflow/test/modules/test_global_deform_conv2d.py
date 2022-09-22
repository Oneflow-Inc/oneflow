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
import oneflow.unittest
import torch as pytorch
import torchvision
from oneflow.test_utils.automated_test_util import *


def _test_deform_conv2d(test_case, placement):
    input_sbp = random_sbp(placement, max_dim=4)
    input_dims = [8, 8, 8, 8]
    input = random_tensor(4, *input_dims).to_global(placement=placement, sbp=input_sbp)

    offset_sbp = random_sbp(placement, max_dim=2)
    offset_dims = [8, 32, 5, 5]
    offset = random_tensor(4, *offset_dims).to_global(
        placement=placement, sbp=offset_sbp
    )

    mask_sbp = random_sbp(placement, max_dim=2)
    mask_dims = [8, 4 * 4, 5, 5]
    mask = random_tensor(4, *mask_dims).to_global(placement=placement, sbp=mask_sbp)

    weight_sbp = random_sbp(placement, max_dim=2)
    weight_dims = [8, 8, 4, 4]
    weight = random_tensor(4, *weight_dims).to_global(
        placement=placement, sbp=weight_sbp
    )

    bias_sbp = random_sbp(placement, max_dim=1)
    bias_dims = [8]
    bias = random_tensor(1, *bias_dims).to_global(placement=placement, sbp=bias_sbp)

    flow_input = input.oneflow.detach().requires_grad_()
    torch_input = input.pytorch.detach().requires_grad_()
    flow_offset = offset.oneflow.detach().requires_grad_()
    torch_offset = offset.pytorch.detach().requires_grad_()
    flow_weight = weight.oneflow.detach().requires_grad_()
    torch_weight = weight.pytorch.detach().requires_grad_()
    flow_mask = mask.oneflow.detach().requires_grad_()
    torch_mask = mask.pytorch.detach().requires_grad_()
    flow_bias = bias.oneflow.detach().requires_grad_()
    torch_bias = bias.pytorch.detach().requires_grad_()

    torch_out = torchvision.ops.deform_conv2d(
        torch_input, torch_offset, torch_weight, mask=torch_mask, bias=torch_bias
    )
    flow_out = oneflow.nn.functional.deform_conv2d(
        flow_input, flow_offset, flow_weight, mask=flow_mask, bias=flow_bias
    )

    # compare forward
    test_case.assertTrue(
        np.allclose(
            flow_out.numpy(), torch_out.detach().cpu().numpy(), rtol=1e-04, atol=1e-4
        )
    )

    # compare backward
    flow_out.sum().backward()
    torch_out.sum().backward()

    flow_input_grad = flow_input.grad
    torch_input_grad = torch_input.grad.detach().cpu()
    flow_weight_grad = flow_weight.grad
    torch_weight_grad = torch_weight.grad.detach().cpu()
    flow_offset_grad = flow_offset.grad
    torch_offset_grad = torch_offset.grad.detach().cpu()
    flow_mask_grad = flow_mask.grad
    torch_mask_grad = torch_mask.grad.detach().cpu()
    flow_bias_grad = flow_bias.grad
    torch_bias_grad = torch_bias.grad.detach().cpu()

    test_case.assertTrue(
        np.allclose(
            flow_input_grad.numpy(), torch_input_grad.numpy(), rtol=1e-04, atol=1e-4
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_weight_grad.numpy(), torch_weight_grad.numpy(), rtol=1e-04, atol=1e-4
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_offset_grad.numpy(), torch_offset_grad.numpy(), rtol=1e-04, atol=1e-4
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_mask_grad.numpy(), torch_mask_grad.numpy(), rtol=1e-04, atol=1e-4
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_bias_grad.numpy(), torch_bias_grad.numpy(), rtol=1e-04, atol=1e-4
        )
    )


class TestGlobalDeformConv2d(flow.unittest.TestCase):
    @globaltest
    def test_deform_conv2d(test_case):
        for placement in all_placement():
            for count in range(5):
                _test_deform_conv2d(test_case, placement)


if __name__ == "__main__":
    unittest.main()
