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
from automated_test_util import *
from test_util import GenArgList

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
    return np.prod(tu).astype(np.int).item()


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
    test_case.assertTrue(
        np.allclose(
            zero_point_of.astype(np.int), zero_point_np.astype(np.int), rtol=0.001
        )
    )


def _run_test_min_max_observer(
    test_case,
    device_type,
    device_num,
    dtype,
    weight_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    per_layer_quantization,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        flow.config.gpu_device_num(device_num)

    @flow.global_function(type="predict", function_config=flow.FunctionConfig())
    def QuantizeJob(
        weight: oft.Numpy.Placeholder(weight_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device_type, "0:0-%d" % (device_num - 1)):
            (scale, zero_point) = flow.quantization.min_max_observer(
                weight,
                quantization_bit,
                quantization_scheme,
                quantization_formula,
                per_layer_quantization,
            )
        return (scale, zero_point)

    weight = (np.random.random(weight_shape) - 0.5).astype(type_name_to_np_type[dtype])
    (scale, zero_point) = QuantizeJob(weight).get()
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


class TestFakeQuantization(flow.unittest.TestCase):
    def test_flip(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = []
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
