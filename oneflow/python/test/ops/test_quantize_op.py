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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft


def gen_quant_scale_per_layer_symmetric(weight, quantize_to_bit):
    weight_max = np.max(np.abs(weight))
    denominator = 2.0 ** (quantize_to_bit - 1) - 1
    return weight_max / denominator, 0


def gen_quant_scale_per_layer_affine(weight, quantize_to_bit):
    weight_max = np.max(weight)
    weight_min = np.min(weight)
    denominator = 2.0 ** (quantize_to_bit) - 1
    scale = (weight_max - weight_min) / denominator
    zero_point = -weight_min / scale
    return scale, zero_point


def product(tu):
    p = 1
    for t in tu:
        p = p * t
    return p


def _check(
    test_case,
    weight,
    scale_of,
    zero_point_of,
    quantize_to_bit,
    quantizer_type,
    per_layer_quantization,
):
    if per_layer_quantization:
        outer_num = 1
        inner_num = product(weight.shape[0:])
    else:
        outer_num = weight.shape[0]
        inner_num = product(weight.shape[1:])

    scale_np = np.zeros((outer_num,))
    zero_point_np = np.zeros((outer_num,))

    weight_flatten = weight.flatten()

    if quantizer_type == "symmetric":
        for c in range(outer_num):
            scale_np[c], zero_point_np[c] = gen_quant_scale_per_layer_symmetric(
                weight_flatten[c * inner_num : (c + 1) * inner_num], quantize_to_bit
            )
    else:  # "affine"
        for c in range(outer_num):
            scale_np[c], zero_point_np[c] = gen_quant_scale_per_layer_affine(
                weight_flatten[c * inner_num : (c + 1) * inner_num], quantize_to_bit
            )

    # print(weight)
    print(scale_of, zero_point_of)
    print(scale_np, zero_point_np)

    test_case.assertTrue(np.allclose(scale_of, scale_np, rtol=1e-3))
    test_case.assertTrue(
        np.allclose(
            zero_point_of.astype(np.int), zero_point_np.astype(np.int), rtol=1e-3
        )
    )


def _run_test(
    test_case,
    device_type,
    dtype,
    weight_shape,
    quantize_to_bit,
    quantizer_type,
    per_layer_quantization,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.enable_debug_mode(True)

    @flow.global_function(type="predict", function_config=flow.FunctionConfig())
    def QuantizeJob(
        weight: oft.Numpy.Placeholder(weight_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device_type, "0:0"):
            scale, zero_point = flow.nn.generate_quantize_scale_for_weight(
                weight, quantize_to_bit, quantizer_type, per_layer_quantization
            )
        return scale, zero_point

    check_point = flow.train.CheckPoint()
    check_point.init()
    weight = (np.random.random(weight_shape) - 1).astype(type_name_to_np_type[dtype])
    scale, zero_point = QuantizeJob(weight).get()

    _check(
        test_case,
        weight,
        scale.numpy(),
        zero_point.numpy(),
        quantize_to_bit,
        quantizer_type,
        per_layer_quantization,
    )


@flow.unittest.skip_unless_1n1d()
class TestGenQuantScaleForWeight(flow.unittest.TestCase):
    def test_gen_quant_scale_for_weight(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["weight_shape"] = [(10, 10, 20, 20), (10, 3, 3, 3), (9, 10, 20, 20)]
        arg_dict["quantize_to_bit"] = [8, 7, 6, 5, 4, 3, 2]
        arg_dict["quantizer_type"] = ["symmetric", "affine"]
        arg_dict["per_layer_quantization"] = [True, False]

        for arg in GenArgList(arg_dict):
            print(arg)
            _run_test(*arg)


if __name__ == "__main__":
    unittest.main()
