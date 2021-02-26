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
from collections import OrderedDict
import math
import numpy as np
import unittest

import oneflow as flow
import oneflow.typing as oft
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def gen_quant_scale_for_min_max_symmetric(weight, quantization_bit):
    weight_max = np.max(np.abs(weight))
    denominator = 2.0 ** (quantization_bit - 1) - 1
    return weight_max / denominator, 0


def gen_quant_scale_for_min_max_affine(weight, quantization_bit):
    weight_max = np.max(weight)
    weight_min = np.min(weight)
    denominator = 2.0 ** (quantization_bit) - 1
    scale = (weight_max - weight_min) / denominator
    zero_point = -weight_min / scale
    return scale, zero_point


def gen_quant_scale_for_min_max_cambricon(weight, quantization_bit):
    weight_max = np.max(np.abs(weight))
    scale = math.floor(math.log2(weight_max)) - (quantization_bit - 2)
    return scale, 0


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
                (
                    scale_np[c],
                    zero_point_np[c],
                ) = gen_quant_scale_for_min_max_symmetric(
                    weight_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                )
        else:  # "affine"
            for c in range(outer_num):
                scale_np[c], zero_point_np[c] = gen_quant_scale_for_min_max_affine(
                    weight_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                )
    else:  # quantization_formula == "cambricon"
        scale_np[0], zero_point_np[0] = gen_quant_scale_for_min_max_cambricon(
            weight_flatten, quantization_bit
        )
    test_case.assertTrue(np.allclose(scale_of, scale_np, rtol=1e-3))
    test_case.assertTrue(
        np.allclose(
            zero_point_of.astype(np.int), zero_point_np.astype(np.int), rtol=1e-3
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
            scale, zero_point = flow.quantization.min_max_observer(
                weight,
                quantization_bit,
                quantization_scheme,
                quantization_formula,
                per_layer_quantization,
            )
        return scale, zero_point

    weight = (np.random.random(weight_shape) - 0.5).astype(type_name_to_np_type[dtype])
    scale, zero_point = QuantizeJob(weight).get()
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


def gen_quant_scale_for_moving_average_min_max_symmetric(
    activation, quantization_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(np.abs(activation))

    denominator = 2.0 ** (quantization_bit - 1) - 1

    if moving_max[0] == 0:
        moving_max[0] = activation_max
    else:
        moving_max[0] = moving_max[0] * momentum + activation_max * (1 - momentum)

    moving_min[0] = moving_max[0]

    return moving_max[0] / denominator, 0


def gen_quant_scale_for_moving_average_min_max_affine(
    activation, quantization_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(activation)
    activation_min = np.min(activation)

    denominator = 2.0 ** (quantization_bit) - 1

    if moving_max[0] == 0:
        moving_max[0] = activation_max
    else:
        moving_max[0] = moving_max[0] * momentum + activation_max * (1 - momentum)

    if moving_min[0] == 0:
        moving_min[0] = activation_min
    else:
        moving_min[0] = moving_min[0] * momentum + activation_min * (1 - momentum)

    scale = (moving_max[0] - moving_min[0]) / denominator
    zero_point = -moving_min[0] / scale

    return scale, zero_point


def gen_quant_scale_for_moving_average_min_max_cambricon(
    activation, quantization_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(np.abs(activation))

    if moving_max[0] == 0:
        moving_max[0] = activation_max
    else:
        moving_max[0] = moving_max[0] * momentum + activation_max * (1 - momentum)

    moving_min[0] = moving_max[0]

    return math.floor(math.log2(moving_max[0])) - (quantization_bit - 2), 0


def _check_moving_average_min_max_observer(
    test_case,
    activation,
    scale_of,
    zero_point_of,
    moving_max_np,
    moving_min_np,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    momentum,
):
    if quantization_formula == "google":
        if quantization_scheme == "symmetric":
            (
                scale_np,
                zero_point_np,
            ) = gen_quant_scale_for_moving_average_min_max_symmetric(
                activation.flatten(),
                quantization_bit,
                momentum,
                moving_max_np,
                moving_min_np,
            )
        else:  # "affine"
            scale_np, zero_point_np = gen_quant_scale_for_moving_average_min_max_affine(
                activation.flatten(),
                quantization_bit,
                momentum,
                moving_max_np,
                moving_min_np,
            )
    else:  # quantization_formula == "cambricon":
        scale_np, zero_point_np = gen_quant_scale_for_moving_average_min_max_cambricon(
            activation.flatten(),
            quantization_bit,
            momentum,
            moving_max_np,
            moving_min_np,
        )
    test_case.assertTrue(np.allclose(scale_of[0], scale_np, rtol=1e-3))
    test_case.assertTrue(np.allclose(zero_point_of[0], zero_point_np, rtol=1e-3))


def _run_test_moving_average_min_max_observer(
    test_case,
    device_type,
    device_num,
    dtype,
    activation_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    momentum,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        flow.config.gpu_device_num(device_num)

    @flow.global_function(type="train", function_config=flow.FunctionConfig())
    def QuantizeJob(
        activation: oft.Numpy.Placeholder(
            activation_shape, dtype=type_name_to_flow_type[dtype]
        )
    ):
        with flow.scope.placement(device_type, "0:0-%d" % (device_num - 1)):
            x = flow.get_variable(
                "x",
                shape=activation_shape,
                dtype=activation.dtype,
                initializer=flow.zeros_initializer(activation.dtype),
                trainable=True,
            )
            scale, zero_point = flow.quantization.moving_average_min_maxObserver(
                activation,
                quantization_bit,
                quantization_scheme,
                quantization_formula,
                momentum,
            )
            fake = x + activation
            loss = flow.math.reduce_mean(fake)
            flow.optimizer.Adam(
                flow.optimizer.PiecewiseConstantScheduler([], [0.001]),
            ).minimize(loss)
        return scale, zero_point

    moving_max_np = np.zeros((1,))
    moving_min_np = np.zeros((1,))

    for i in range(10):
        activation = (np.random.random(activation_shape) - 0.5).astype(
            type_name_to_np_type[dtype]
        )
        scale, zero_point = QuantizeJob(activation).get()
        _check_moving_average_min_max_observer(
            test_case,
            activation,
            scale.numpy(),
            zero_point.numpy(),
            moving_max_np,
            moving_min_np,
            quantization_bit,
            quantization_scheme,
            quantization_formula,
            momentum,
        )


def fake_quant_per_layer_symmetric(input, quantization_bit, scale):
    upper_bound = 2.0 ** (quantization_bit - 1) - 1
    lower_bound = -upper_bound
    return np.clip(np.rint(input / scale), lower_bound, upper_bound) * scale


def fake_quant_per_layer_affine(input, quantization_bit, scale, zero_point):
    upper_bound = 2.0 ** (quantization_bit) - 1
    lower_bound = 0
    return (
        np.clip(np.rint(input / scale + zero_point), lower_bound, upper_bound)
        - zero_point
    ) * scale


def fake_quant_per_layer_cambricon(input, quantization_bit, shift):
    upper_bound = 2.0 ** (quantization_bit - 1) - 1
    lower_bound = -upper_bound
    scale = 2 ** shift
    return np.clip(np.rint(input / scale), lower_bound, upper_bound) * scale


def _check_fake_quantize(
    test_case,
    input,
    input_diff_of,
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
                (
                    scale_np[c],
                    zero_point_np[c],
                ) = gen_quant_scale_for_min_max_symmetric(
                    input_flatten[c * inner_num : (c + 1) * inner_num], quantization_bit
                )
                out = fake_quant_per_layer_symmetric(
                    input_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                    scale_np[c],
                )
                out_np[c * inner_num : (c + 1) * inner_num] = out

        else:  # "affine"
            for c in range(outer_num):
                scale_np[c], zero_point_np[c] = gen_quant_scale_for_min_max_affine(
                    input_flatten[c * inner_num : (c + 1) * inner_num], quantization_bit
                )
                out = fake_quant_per_layer_affine(
                    input_flatten[c * inner_num : (c + 1) * inner_num],
                    quantization_bit,
                    scale_np[c],
                    zero_point_np[c],
                )
                out_np[c * inner_num : (c + 1) * inner_num] = out
    else:  # quantization_formula == "cambricon"
        scale_np[0], zero_point_np[0] = gen_quant_scale_for_min_max_cambricon(
            input_flatten, quantization_bit
        )
        out_np = fake_quant_per_layer_cambricon(
            input_flatten, quantization_bit, scale_np[0]
        )

    # NOTE(Liang Depeng):
    # The slightly different rounding results between C++ and Python will make
    # the dequantize results very differently. So enlarge the tolerant to
    # avoid the test failure.
    test_case.assertTrue(np.mean(np.abs(out_of - out_np)) < 1e-5)
    test_case.assertTrue(np.allclose(input_diff_of, input_diff_np, rtol=1e-3))


def _run_test_fake_quantize(
    test_case,
    device_type,
    device_num,
    dtype,
    in_shape,
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

    @flow.global_function(type="train", function_config=flow.FunctionConfig())
    def QuantizeJob(
        input: oft.Numpy.Placeholder(in_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=in_shape,
                dtype=input.dtype,
                initializer=flow.zeros_initializer(input.dtype),
                trainable=True,
            )
            input_x = input + x

        flow.watch_diff(input_x, test_global_storage.Setter("input_diff"))

        with flow.scope.placement(device_type, "0:0-%d" % (device_num - 1)):
            scale, zero_point = flow.quantization.min_max_observer(
                input_x,
                quantization_bit,
                quantization_scheme,
                quantization_formula,
                per_layer_quantization,
            )
            out = flow.quantization.fake_quantization(
                input_x,
                scale,
                zero_point,
                quantization_bit,
                quantization_scheme,
                quantization_formula,
            )
            loss = flow.math.reduce_mean(out)

            flow.optimizer.Adam(
                flow.optimizer.PiecewiseConstantScheduler([], [0.001]),
            ).minimize(loss)

        return out

    input = (np.random.random(in_shape) - 0.5).astype(type_name_to_np_type[dtype])
    out = QuantizeJob(input).get()

    input_diff = test_global_storage.Get("input_diff")

    _check_fake_quantize(
        test_case,
        input,
        input_diff.flatten(),
        out.numpy().flatten(),
        quantization_bit,
        quantization_scheme,
        quantization_formula,
        per_layer_quantization,
    )


@flow.unittest.skip_unless_1n4d()
class TestMinMaxObserver(flow.unittest.TestCase):
    def test_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["device_num"] = [1, 4]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["weight_shape"] = [(89, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google", "cambricon"]
        arg_dict["per_layer_quantization"] = [True, False]

        for arg in GenArgList(arg_dict):
            _run_test_min_max_observer(*arg)


@flow.unittest.skip_unless_1n4d()
class TestMovingAverageMinMaxObserver(flow.unittest.TestCase):
    def test_moving_average_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["device_num"] = [1, 4]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["activation_shape"] = [(89, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google", "cambricon"]
        arg_dict["momentum"] = [0.95]

        for arg in GenArgList(arg_dict):
            _run_test_moving_average_min_max_observer(*arg)


@flow.unittest.skip_unless_1n4d()
class TestFakeQuantize(flow.unittest.TestCase):
    def test_fake_quantize(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["device_num"] = [1, 4]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["in_shape"] = [(89, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google", "cambricon"]
        arg_dict["per_layer_quantization"] = [True, False]

        for arg in GenArgList(arg_dict):
            _run_test_fake_quantize(*arg)


if __name__ == "__main__":
    unittest.main()
