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
    atol = 0.8

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

    conv1d = flow.nn.Conv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    np_rand = np.random.rand(batch_size, input_channels, spatial_size)
    qat_input = flow.tensor(
        np_rand, dtype=flow.float32, requires_grad=True, device=device
    )
    normal_input = flow.tensor(
        np_rand, dtype=flow.float32, requires_grad=True, device=device
    )

    qat_out = qat_conv1d(qat_input)
    out = conv1d(normal_input)

    cosine_distance = flow.nn.functional.cosine_similarity(
        qat_out.flatten(), out.flatten(), dim=0
    )
    test_case.assertTrue(cosine_distance.numpy() > atol)

    qat_out.sum().backward()
    out.sum().backward()

    cosine_distance = flow.nn.functional.cosine_similarity(
        qat_input.grad.flatten(), normal_input.grad.flatten(), dim=0
    )
    test_case.assertTrue(cosine_distance.numpy() > atol)


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
    atol = 0.8

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

    conv2d = flow.nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    np_rand = np.random.rand(batch_size, input_channels, spatial_size, spatial_size)
    qat_input = flow.tensor(
        np_rand, dtype=flow.float32, requires_grad=True, device=device
    )
    normal_input = flow.tensor(
        np_rand, dtype=flow.float32, requires_grad=True, device=device
    )

    qat_out = qat_conv2d(qat_input)
    out = conv2d(normal_input)

    cosine_distance = flow.nn.functional.cosine_similarity(
        qat_out.flatten(), out.flatten(), dim=0
    )
    test_case.assertTrue(cosine_distance.numpy() > atol)

    qat_out.sum().backward()
    out.sum().backward()

    cosine_distance = flow.nn.functional.cosine_similarity(
        qat_input.grad.flatten(), normal_input.grad.flatten(), dim=0
    )
    test_case.assertTrue(cosine_distance.numpy() > atol)


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
    atol = 0.8

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

    conv3d = flow.nn.Conv3d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    np_rand = np.random.rand(
        batch_size, input_channels, spatial_size, spatial_size, spatial_size
    )
    qat_input = flow.tensor(
        np_rand, dtype=flow.float32, requires_grad=True, device=device
    )
    normal_input = flow.tensor(
        np_rand, dtype=flow.float32, requires_grad=True, device=device
    )

    qat_out = qat_conv3d(qat_input)
    out = conv3d(normal_input)

    cosine_distance = flow.nn.functional.cosine_similarity(
        qat_out.flatten(), out.flatten(), dim=0
    )
    test_case.assertTrue(cosine_distance.numpy() > atol)

    qat_out.sum().backward()
    out.sum().backward()

    cosine_distance = flow.nn.functional.cosine_similarity(
        qat_input.grad.flatten(), normal_input.grad.flatten(), dim=0
    )
    test_case.assertTrue(cosine_distance.numpy() > atol)


@flow.unittest.skip_unless_1n1d()
class TestQatModules(flow.unittest.TestCase):
    def test_qat_conv1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["quantization_bit"] = [4, 8]
        arg_dict["quantization_scheme"] = ["symmetric"]
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
        arg_dict["quantization_scheme"] = ["symmetric"]
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
        arg_dict["quantization_scheme"] = ["symmetric"]
        arg_dict["weight_quant_per_layer"] = [True, False]
        arg_dict["input_quant_momentum"] = [0.95]

        for i in range(5):
            for arg in GenArgList(arg_dict):
                _test_qat_conv3d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
