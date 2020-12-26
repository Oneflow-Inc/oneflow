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
import numpy as np
import unittest

import oneflow as flow
import oneflow.typing as oft
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def gen_quant_scale_for_min_max_symmetric(weight, quantize_to_bit):
    weight_max = np.max(np.abs(weight))
    denominator = 2.0 ** (quantize_to_bit - 1) - 1
    return weight_max / denominator, 0


def gen_quant_scale_for_min_max_affine(weight, quantize_to_bit):
    weight_max = np.max(weight)
    weight_min = np.min(weight)
    denominator = 2.0 ** (quantize_to_bit) - 1
    scale = (weight_max - weight_min) / denominator
    zero_point = -weight_min / scale
    return scale, zero_point


def product(tu):
    return np.prod(tu).astype(np.int).item()


def _check_min_max_observer(
    test_case,
    weight,
    scale_of,
    zero_point_of,
    quantize_to_bit,
    quantize_scheme,
    per_layer_quantize,
):
    if per_layer_quantize:
        outer_num = 1
        inner_num = product(weight.shape[0:])
    else:
        outer_num = weight.shape[0]
        inner_num = product(weight.shape[1:])

    scale_np = np.zeros((outer_num,))
    zero_point_np = np.zeros((outer_num,))

    weight_flatten = weight.flatten()

    if quantize_scheme == "symmetric":
        for c in range(outer_num):
            (scale_np[c], zero_point_np[c],) = gen_quant_scale_for_min_max_symmetric(
                weight_flatten[c * inner_num : (c + 1) * inner_num], quantize_to_bit
            )
    else:  # "affine"
        for c in range(outer_num):
            scale_np[c], zero_point_np[c] = gen_quant_scale_for_min_max_affine(
                weight_flatten[c * inner_num : (c + 1) * inner_num], quantize_to_bit
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
    dtype,
    weight_shape,
    quantize_to_bit,
    quantize_scheme,
    per_layer_quantize,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    @flow.global_function(type="predict", function_config=flow.FunctionConfig())
    def QuantizeJob(
        weight: oft.Numpy.Placeholder(weight_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device_type, "0:0"):
            scale, zero_point = flow.quantization.MinMaxObserver(
                weight, quantize_to_bit, quantize_scheme, per_layer_quantize
            )
        return scale, zero_point

    check_point = flow.train.CheckPoint()
    check_point.init()
    weight = (np.random.random(weight_shape) - 0.5).astype(type_name_to_np_type[dtype])
    scale, zero_point = QuantizeJob(weight).get()

    _check_min_max_observer(
        test_case,
        weight,
        scale.numpy(),
        zero_point.numpy(),
        quantize_to_bit,
        quantize_scheme,
        per_layer_quantize,
    )


def gen_quant_scale_for_moving_average_min_max_symmetric(
    activation, quantize_to_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(np.abs(activation))

    denominator = 2.0 ** (quantize_to_bit - 1) - 1

    if moving_max[0] == 0:
        moving_max[0] = activation_max
    else:
        moving_max[0] = moving_max[0] * momentum + activation_max * (1 - momentum)

    moving_min[0] = moving_max[0]

    return moving_max[0] / denominator, 0


def gen_quant_scale_for_moving_average_min_max_affine(
    activation, quantize_to_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(activation)
    activation_min = np.min(activation)

    denominator = 2.0 ** (quantize_to_bit) - 1

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


def _check_moving_average_min_max_observer(
    test_case,
    activation,
    scale_of,
    zero_point_of,
    moving_max_np,
    moving_min_np,
    quantize_to_bit,
    quantize_scheme,
    momentum,
):
    if quantize_scheme == "symmetric":
        scale_np, zero_point_np = gen_quant_scale_for_moving_average_min_max_symmetric(
            activation.flatten(),
            quantize_to_bit,
            momentum,
            moving_max_np,
            moving_min_np,
        )
    else:  # "affine"
        scale_np, zero_point_np = gen_quant_scale_for_moving_average_min_max_affine(
            activation.flatten(),
            quantize_to_bit,
            momentum,
            moving_max_np,
            moving_min_np,
        )

    test_case.assertTrue(np.allclose(scale_of[0], scale_np, rtol=1e-3))
    test_case.assertTrue(np.allclose(zero_point_of[0], zero_point_np, rtol=1e-3))


def _run_test_moving_average_min_max_observer(
    test_case,
    device_type,
    dtype,
    activation_shape,
    quantize_to_bit,
    quantize_scheme,
    momentum,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    @flow.global_function(type="predict", function_config=flow.FunctionConfig())
    def QuantizeJob(
        activation: oft.Numpy.Placeholder(
            activation_shape, dtype=type_name_to_flow_type[dtype]
        )
    ):
        with flow.scope.placement(device_type, "0:0"):
            moving_max = flow.get_variable(
                "moving_max",
                shape=(1,),
                dtype=activation.dtype,
                initializer=flow.zeros_initializer(activation.dtype),
                trainable=False,
            )
            moving_min = flow.get_variable(
                "moving_min",
                shape=(1,),
                dtype=activation.dtype,
                initializer=flow.zeros_initializer(activation.dtype),
                trainable=False,
            )
            scale, zero_point = flow.quantization.MovingAverageMinMaxObserver(
                activation,
                moving_max,
                moving_min,
                quantize_to_bit,
                quantize_scheme,
                momentum,
            )
            return scale, zero_point

    check_point = flow.train.CheckPoint()
    check_point.init()

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
            quantize_to_bit,
            quantize_scheme,
            momentum,
        )


def fake_quant_per_layer_symmetric(input, quantize_to_bit, scale):
    upper_bound = 2.0 ** (quantize_to_bit - 1) - 1
    lower_bound = -upper_bound
    return np.clip(np.rint(input / scale), lower_bound, upper_bound) * scale


def fake_quant_per_layer_affine(input, quantize_to_bit, scale, zero_point):
    upper_bound = 2.0 ** (quantize_to_bit) - 1
    lower_bound = 0
    return (
        np.clip(np.rint(input / scale + zero_point), lower_bound, upper_bound)
        - zero_point
    ) * scale


def _check_fake_quantize(
    test_case,
    input,
    scale,
    zero_point,
    input_diff_of,
    out_of,
    quantize_to_bit,
    quantize_scheme,
    per_layer_quantize,
):
    if per_layer_quantize:
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

    if quantize_scheme == "symmetric":
        for c in range(outer_num):
            (scale_np[c], zero_point_np[c],) = gen_quant_scale_for_min_max_symmetric(
                input_flatten[c * inner_num : (c + 1) * inner_num], quantize_to_bit
            )
            out = fake_quant_per_layer_symmetric(
                input_flatten[c * inner_num : (c + 1) * inner_num],
                quantize_to_bit,
                scale_np[c],
            )
            out_np[c * inner_num : (c + 1) * inner_num] = out

    else:  # "affine"
        for c in range(outer_num):
            scale_np[c], zero_point_np[c] = gen_quant_scale_for_min_max_affine(
                input_flatten[c * inner_num : (c + 1) * inner_num], quantize_to_bit
            )
            out = fake_quant_per_layer_affine(
                input_flatten[c * inner_num : (c + 1) * inner_num],
                quantize_to_bit,
                scale_np[c],
                zero_point_np[c],
            )
            out_np[c * inner_num : (c + 1) * inner_num] = out

    # NOTE(Liang Depeng):
    # The slightly different rounding results between C++ and Python will make
    # the dequantize results very differently. So enlarge the tolerant to
    # avoid the test failure.
    test_case.assertTrue(np.mean(out_of - out_np) < 1e-5)
    test_case.assertTrue(np.allclose(input_diff_of, input_diff_np, rtol=1e-3))


def _run_test_fake_quantize(
    test_case,
    device_type,
    dtype,
    in_shape,
    quantize_to_bit,
    quantize_scheme,
    per_layer_quantize,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

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
            input += x
            scale, zero_point = flow.quantization.MinMaxObserver(
                input, quantize_to_bit, quantize_scheme, per_layer_quantize
            )
            out = flow.quantization.FakeQuantize(
                input, scale, zero_point, quantize_to_bit, quantize_scheme
            )
            loss = flow.math.reduce_mean(out)
            flow.optimizer.Adam(
                flow.optimizer.PiecewiseConstantScheduler([], [0.001]),
            ).minimize(loss)

            flow.watch_diff(input, test_global_storage.Setter("input_diff"))

            return out, scale, zero_point

    check_point = flow.train.CheckPoint()
    check_point.init()

    input = (np.random.random(in_shape) - 0.5).astype(type_name_to_np_type[dtype])
    out, scale, zero_point = QuantizeJob(input).get()

    input_diff = test_global_storage.Get("input_diff")

    _check_fake_quantize(
        test_case,
        input,
        scale,
        zero_point,
        input_diff.flatten(),
        out.numpy().flatten(),
        quantize_to_bit,
        quantize_scheme,
        per_layer_quantize,
    )


@flow.unittest.skip_unless_1n1d()
class TestMinMaxObserver(flow.unittest.TestCase):
    def test_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["weight_shape"] = [(89, 40, 20, 10)]
        arg_dict["quantize_to_bit"] = [8, 2]
        arg_dict["quantize_scheme"] = ["symmetric", "affine"]
        arg_dict["per_layer_quantize"] = [True, False]

        for arg in GenArgList(arg_dict):
            _run_test_min_max_observer(*arg)


@flow.unittest.skip_unless_1n1d()
class TestMovingAverageMinMaxObserver(flow.unittest.TestCase):
    def test_moving_average_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["activation_shape"] = [(89, 40, 20, 10)]
        arg_dict["quantize_to_bit"] = [8, 2]
        arg_dict["quantize_scheme"] = ["symmetric", "affine"]
        arg_dict["momentum"] = [0.95]

        for arg in GenArgList(arg_dict):
            _run_test_moving_average_min_max_observer(*arg)


@flow.unittest.skip_unless_1n1d()
class TestFakeQuantize(flow.unittest.TestCase):
    def test_fake_quantize(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["in_shape"] = [(89, 40, 20, 10)]
        arg_dict["quantize_to_bit"] = [8, 2]
        arg_dict["quantize_scheme"] = ["symmetric", "affine"]
        arg_dict["per_layer_quantize"] = [True, False]

        for arg in GenArgList(arg_dict):
            _run_test_fake_quantize(*arg)


if __name__ == "__main__":
    unittest.main()
