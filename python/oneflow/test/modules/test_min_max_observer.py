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
from oneflow.nn.modules import min_max_observer
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


def _check_min_max_observer(
    test_case,
    weight,
    scale_of,
    zero_point_of,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    per_layer_quantization,
):
    if per_layer_quantization or quantization_formula == "cambricon":
        outer_num = 1
        inner_num = product(weight.shape[0:])
    else:
        outer_num = weight.shape[0]
        inner_num = product(weight.shape[1:])
    scale_np = np.zeros((outer_num,))
    zero_point_np = np.zeros((outer_num,))
    weight_flatten = weight.flatten()
    if quantization_formula == "google":
        if quantization_scheme == "symmetric":
            for c in range(outer_num):
                (scale_np[c], zero_point_np[c]) = gen_quant_scale_for_min_max_symmetric(
                    weight_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                )
        else:
            for c in range(outer_num):
                (scale_np[c], zero_point_np[c]) = gen_quant_scale_for_min_max_affine(
                    weight_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                )
    else:
        (scale_np[0], zero_point_np[0]) = gen_quant_scale_for_min_max_cambricon(
            weight_flatten, quantization_bit
        )
    test_case.assertTrue(np.allclose(scale_of, scale_np, rtol=0.001))

    rmse = np.sqrt(np.mean((zero_point_of - zero_point_np) ** 2))
    assert rmse <= 1.0, "min_max_observer op zero_point calculate has bug!"


def _run_test_min_max_observer(
    test_case,
    device_type,
    weight_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    per_layer_quantization,
):
    weight = (np.random.random(weight_shape) - 0.5).astype(np.float32)
    tensor_weight = flow.tensor(
        weight, device=flow.device(device_type), dtype=flow.float32
    )
    min_max_observer = flow.nn.MinMaxObserver(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        per_layer_quantization=per_layer_quantization,
    )
    scale, zero_point = min_max_observer(tensor_weight)
    _check_min_max_observer(
        test_case,
        weight,
        scale.numpy(),
        zero_point.numpy(),
        quantization_bit,
        quantization_scheme,
        quantization_formula,
        per_layer_quantization,
    )


class TestMinMaxObserver(flow.unittest.TestCase):
    def test_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "cuda"]
        arg_dict["weight_shape"] = [(9, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["per_layer_quantization"] = [True, False]
        for arg in GenArgList(arg_dict):
            if arg[-2] == "cambricon" and arg[-1] == False:
                continue
            _run_test_min_max_observer(*arg)


if __name__ == "__main__":
    unittest.main()
