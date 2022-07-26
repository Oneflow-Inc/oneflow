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
import random
import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_qat_fuse_conv_bn_1d(
    test_case,
    device,
    quantization_formula,
    quantization_bit,
    quantization_scheme,
    weight_per_layer_quantization,
    input_quant_momentum,
    bn_affine,
):
    batch_size = random.randint(1, 5)
    input_channels = random.randint(1, 3)
    output_channels = random.randint(1, 3)
    spatial_size = random.randint(8, 16)
    kernel_size = random.randint(1, 3)
    stride = random.randint(1, 2)
    padding = random.randint(0, 2)

    conv1d_for_qat = flow.nn.Conv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    bn1d_for_qat = flow.nn.BatchNorm1d(output_channels, affine=bn_affine,).to(device)

    fused_conv_bn_1d = flow.nn.QatFuseConvBN(
        conv1d_for_qat,
        bn1d_for_qat,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        weight_per_layer_quantization=weight_per_layer_quantization,
        input_quant_momentum=input_quant_momentum,
    ).to(device)

    qat_input = flow.rand(
        batch_size,
        input_channels,
        spatial_size,
        dtype=flow.float32,
        requires_grad=True,
        device=device,
    )

    qat_out = fused_conv_bn_1d(qat_input)
    qat_out.sum().backward()
    qat_out.numpy()
    qat_input.grad.numpy()

    # check correctness after freeze
    fused_conv_bn_1d.eval()
    qat_out_before_freeze = fused_conv_bn_1d(qat_input)
    flow.nn.qat.freeze_all_qat_submodules(fused_conv_bn_1d)
    qat_out_after_freeze = fused_conv_bn_1d(qat_input)
    test_case.assertTrue(
        np.allclose(qat_out_before_freeze.numpy(), qat_out_after_freeze.numpy(), atol=1e-6)
    )


def _test_qat_fuse_conv_bn_2d(
    test_case,
    device,
    quantization_formula,
    quantization_bit,
    quantization_scheme,
    weight_per_layer_quantization,
    input_quant_momentum,
    bn_affine,
):
    batch_size = random.randint(1, 5)
    input_channels = random.randint(1, 3)
    output_channels = random.randint(2, 3)
    spatial_size = random.randint(8, 16)
    kernel_size = random.randint(1, 3)
    stride = random.randint(1, 2)
    padding = random.randint(0, 2)

    conv2d_for_qat = flow.nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    bn2d_for_qat = flow.nn.BatchNorm2d(output_channels, affine=bn_affine,).to(device)
    fused_conv_bn_2d = flow.nn.QatFuseConvBN(
        conv2d_for_qat,
        bn2d_for_qat,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        weight_per_layer_quantization=weight_per_layer_quantization,
        input_quant_momentum=input_quant_momentum,
    ).to(device)

    qat_input = flow.rand(
        batch_size,
        input_channels,
        spatial_size,
        spatial_size,
        dtype=flow.float32,
        requires_grad=True,
        device=device,
    )
    qat_out = fused_conv_bn_2d(qat_input)
    qat_out.sum().backward()
    qat_out.numpy()
    qat_input.grad.numpy()

    # check correctness after freeze
    fused_conv_bn_2d.eval()
    qat_out_before_freeze = fused_conv_bn_2d(qat_input)
    flow.nn.qat.freeze_all_qat_submodules(fused_conv_bn_2d)
    qat_out_after_freeze = fused_conv_bn_2d(qat_input)
    test_case.assertTrue(
        np.allclose(qat_out_before_freeze.numpy(), qat_out_after_freeze.numpy(), atol=1e-6)
    )


def _test_qat_fuse_conv_bn_3d(
    test_case,
    device,
    quantization_formula,
    quantization_bit,
    quantization_scheme,
    weight_per_layer_quantization,
    input_quant_momentum,
    bn_affine,
):
    batch_size = random.randint(1, 5)
    input_channels = random.randint(1, 3)
    output_channels = random.randint(2, 3)
    spatial_size = random.randint(8, 16)
    kernel_size = random.randint(1, 3)
    stride = random.randint(1, 2)
    padding = random.randint(0, 2)

    conv3d_for_qat = flow.nn.Conv3d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    bn3d_for_qat = flow.nn.BatchNorm3d(output_channels, affine=bn_affine,).to(device)

    fused_conv_bn_3d = flow.nn.QatFuseConvBN(
        conv3d_for_qat,
        bn3d_for_qat,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        weight_per_layer_quantization=weight_per_layer_quantization,
        input_quant_momentum=input_quant_momentum,
    ).to(device)

    qat_input = flow.rand(
        batch_size,
        input_channels,
        spatial_size,
        spatial_size,
        spatial_size,
        dtype=flow.float32,
        requires_grad=True,
        device=device,
    )

    qat_out = fused_conv_bn_3d(qat_input)
    qat_out.sum().backward()
    qat_out.numpy()
    qat_input.grad.numpy()

    # check correctness after freeze
    fused_conv_bn_3d.eval()
    qat_out_before_freeze = fused_conv_bn_3d(qat_input)
    flow.nn.qat.freeze_all_qat_submodules(fused_conv_bn_3d)
    qat_out_after_freeze = fused_conv_bn_3d(qat_input)
    test_case.assertTrue(
        np.allclose(qat_out_before_freeze.numpy(), qat_out_after_freeze.numpy(), atol=1e-6)
    )


@flow.unittest.skip_unless_1n1d()
class TestQatFuseConvBNModules(flow.unittest.TestCase):
    def test_qat_fuse_conv_bn_1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric"]
        arg_dict["weight_per_layer_quantization"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]
        arg_dict["bn_affine"] = [True, False]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_fuse_conv_bn_1d(test_case, *arg)

    def test_qat_fuse_conv_bn_2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric"]
        arg_dict["weight_per_layer_quantization"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]
        arg_dict["bn_affine"] = [True, False]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_fuse_conv_bn_2d(test_case, *arg)

    def test_qat_fuse_conv_bn_3d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric"]
        arg_dict["weight_per_layer_quantization"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]
        arg_dict["bn_affine"] = [True, False]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_fuse_conv_bn_3d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
