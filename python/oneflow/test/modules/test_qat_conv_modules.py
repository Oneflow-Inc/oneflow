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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_qat_conv1d(
    test_case,
    device,
    quantization_formula,
    quantization_bit,
    quantization_scheme,
    weight_quant_per_layer,
    input_quant_momentum,
):
    batch_size = random.randint(1, 5)
    input_channels = random.randint(1, 3)
    output_channels = random.randint(1, 3)
    spatial_size = random.randint(8, 16)
    kernel_size = random.randint(1, 3)
    stride = random.randint(1, 2)
    padding = random.randint(0, 2)

    qat_conv1d = flow.nn.QatConv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        weight_quant_per_layer=weight_quant_per_layer,
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

    qat_out = qat_conv1d(qat_input)
    qat_out.sum().backward()
    qat_out.numpy()
    qat_input.grad.numpy()


def _test_qat_conv2d(
    test_case,
    device,
    quantization_formula,
    quantization_bit,
    quantization_scheme,
    weight_quant_per_layer,
    input_quant_momentum,
):
    batch_size = random.randint(1, 5)
    input_channels = random.randint(1, 3)
    output_channels = random.randint(1, 3)
    spatial_size = random.randint(8, 16)
    kernel_size = random.randint(1, 3)
    stride = random.randint(1, 2)
    padding = random.randint(0, 2)

    qat_conv2d = flow.nn.QatConv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        weight_quant_per_layer=weight_quant_per_layer,
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
    qat_out = qat_conv2d(qat_input)
    qat_out.sum().backward()
    qat_out.numpy()
    qat_input.grad.numpy()


def _test_qat_conv3d(
    test_case,
    device,
    quantization_formula,
    quantization_bit,
    quantization_scheme,
    weight_quant_per_layer,
    input_quant_momentum,
):
    batch_size = random.randint(1, 5)
    input_channels = random.randint(1, 3)
    output_channels = random.randint(1, 3)
    spatial_size = random.randint(8, 16)
    kernel_size = random.randint(1, 3)
    stride = random.randint(1, 2)
    padding = random.randint(0, 2)

    qat_conv3d = flow.nn.QatConv3d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        weight_quant_per_layer=weight_quant_per_layer,
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
    qat_out = qat_conv3d(qat_input)
    qat_out.sum().backward()
    qat_out.numpy()
    qat_input.grad.numpy()


@flow.unittest.skip_unless_1n1d()
class TestQatModules(flow.unittest.TestCase):
    def test_qat_conv1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["weight_quant_per_layer"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_conv1d(test_case, *arg)

    def test_qat_conv2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["weight_quant_per_layer"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_conv2d(test_case, *arg)

    def test_qat_conv3d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["weight_quant_per_layer"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_conv3d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
