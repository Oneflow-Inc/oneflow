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

import math
import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.test_util import (
    GenArgList,
    type_name_to_flow_type,
    type_name_to_np_type,
)

import oneflow as flow
import oneflow.unittest


def gen_quant_scale_for_min_max_symmetric(weight, quantization_bit):
    weight_max = np.max(np.abs(weight))
    denominator = 2.0 ** (quantization_bit - 1) - 1
    return (weight_max / denominator, 0)


def gen_quant_scale_for_min_max_affine(weight, quantization_bit):
    weight_max = np.max(weight)
    weight_min = np.min(weight)
    denominator = 2.0 ** quantization_bit - 1
    scale = (weight_max - weight_min) / denominator
    zero_point = -np.round(weight_min / scale)
    return (scale, zero_point)


def gen_quant_scale_for_min_max_cambricon(weight, quantization_bit):
    weight_max = np.max(np.abs(weight))
    scale = math.floor(math.log2(weight_max)) - (quantization_bit - 2)
    return (scale, 0)


def product(tu):
    return np.prod(tu).astype(np.int32).item()


def quant_per_layer_symmetric(input, quantization_bit, scale):
    upper_bound = 2.0 ** (quantization_bit - 1) - 1
    lower_bound = -upper_bound
    return np.clip(np.rint(input / scale), lower_bound, upper_bound)


def quant_per_layer_affine(input, quantization_bit, scale, zero_point):
    upper_bound = 2.0 ** quantization_bit - 1
    lower_bound = 0
    return np.clip(np.rint(input / scale + zero_point), lower_bound, upper_bound)


def quant_per_layer_cambricon(input, quantization_bit, shift):
    upper_bound = 2.0 ** (quantization_bit - 1) - 1
    lower_bound = -upper_bound
    scale = 2 ** shift
    return np.clip(np.rint(input / scale), lower_bound, upper_bound)


def _check_quantize(
    test_case,
    input,
    out_of,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    per_layer_quantization,
):
    if per_layer_quantization or quantization_formula == "cambricon":
        outer_num = 1
        inner_num = product(input.shape[0:])
    else:
        outer_num = input.shape[0]
        inner_num = product(input.shape[1:])
    scale_np = np.zeros((outer_num,))
    zero_point_np = np.zeros((outer_num,))
    out_np = np.zeros((inner_num * outer_num,))
    input_flatten = input.flatten()
    input_diff_np = np.full((inner_num * outer_num,), 1.0 / (inner_num * outer_num))
    if quantization_formula == "google":
        if quantization_scheme == "symmetric":
            for c in range(outer_num):
                (scale_np[c], zero_point_np[c]) = gen_quant_scale_for_min_max_symmetric(
                    input_flatten[c * inner_num : (c + 1) * inner_num], quantization_bit
                )
                out = quant_per_layer_symmetric(
                    input_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                    scale_np[c],
                )
                out_np[c * inner_num : (c + 1) * inner_num] = out
        else:
            for c in range(outer_num):
                (scale_np[c], zero_point_np[c]) = gen_quant_scale_for_min_max_affine(
                    input_flatten[c * inner_num : (c + 1) * inner_num], quantization_bit
                )
                out = quant_per_layer_affine(
                    input_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                    scale_np[c],
                    zero_point_np[c],
                )
                out_np[c * inner_num : (c + 1) * inner_num] = out
    else:
        (scale_np[0], zero_point_np[0]) = gen_quant_scale_for_min_max_cambricon(
            input_flatten, quantization_bit
        )
        out_np = quant_per_layer_cambricon(input_flatten, quantization_bit, scale_np[0])
    rmse = np.sqrt(np.mean((out_of - out_np) ** 2))
    assert rmse <= 2.0, "quantization op has bug!"


def _run_test_quantize(
    test_case,
    device_type,
    dtype,
    in_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    per_layer_quantization,
):
    input = (np.random.random(in_shape) - 0.5).astype(type_name_to_np_type[dtype])
    input_tensor = flow.tensor(
        input, dtype=flow.float32, device=flow.device(device_type)
    )
    min_max_observer = flow.nn.MinMaxObserver(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        per_layer_quantization=per_layer_quantization,
    )
    (scale, zero_point) = min_max_observer(input_tensor)
    quantization = flow.nn.Quantization(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
    )
    output_tensor = quantization(input_tensor, scale, zero_point)

    out = output_tensor.numpy()
    _check_quantize(
        test_case,
        input,
        out.flatten(),
        quantization_bit,
        quantization_scheme,
        quantization_formula,
        per_layer_quantization,
    )


class TestQuantize(flow.unittest.TestCase):
    def test_quantize(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["in_shape"] = [(9, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["per_layer_quantization"] = [True, False]
        for arg in GenArgList(arg_dict):
            if arg[-2] == "cambricon" and arg[-1] == False:
                continue
            _run_test_quantize(*arg)


if __name__ == "__main__":
    unittest.main()
