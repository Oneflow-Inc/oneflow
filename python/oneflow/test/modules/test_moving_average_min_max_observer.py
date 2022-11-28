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
    return (moving_max[0] / denominator, 0)


def gen_quant_scale_for_moving_average_min_max_affine(
    activation, quantization_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(activation)
    activation_min = np.min(activation)
    denominator = 2.0 ** quantization_bit - 1
    if moving_max[0] == 0:
        moving_max[0] = activation_max
    else:
        moving_max[0] = moving_max[0] * momentum + activation_max * (1 - momentum)
    if moving_min[0] == 0:
        moving_min[0] = activation_min
    else:
        moving_min[0] = moving_min[0] * momentum + activation_min * (1 - momentum)
    scale = (moving_max[0] - moving_min[0]) / denominator
    zero_point = -np.round(moving_min[0] / scale)
    return (scale, zero_point)


def gen_quant_scale_for_moving_average_min_max_cambricon(
    activation, quantization_bit, momentum, moving_max, moving_min
):
    activation_max = np.max(np.abs(activation))
    if moving_max[0] == 0:
        moving_max[0] = activation_max
    else:
        moving_max[0] = moving_max[0] * momentum + activation_max * (1 - momentum)
    moving_min[0] = moving_max[0]
    return (math.floor(math.log2(moving_max[0])) - (quantization_bit - 2), 0)


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
        else:
            (
                scale_np,
                zero_point_np,
            ) = gen_quant_scale_for_moving_average_min_max_affine(
                activation.flatten(),
                quantization_bit,
                momentum,
                moving_max_np,
                moving_min_np,
            )
    else:
        (
            scale_np,
            zero_point_np,
        ) = gen_quant_scale_for_moving_average_min_max_cambricon(
            activation.flatten(),
            quantization_bit,
            momentum,
            moving_max_np,
            moving_min_np,
        )
    test_case.assertTrue(np.allclose(scale_of[0], scale_np, rtol=0.001))

    rmse = np.sqrt(np.mean((zero_point_of[0] - zero_point_np) ** 2))
    assert (
        rmse <= 1.0
    ), "moving_average_min_max_observer op zero_point calculate has bug!"


def _run_test_moving_average_min_max_observer(
    test_case,
    device_type,
    dtype,
    activation_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    momentum,
):
    moving_max_np = np.zeros((1,))
    moving_min_np = np.zeros((1,))
    current_train_step_tensor = flow.tensor(
        np.zeros((1,)).astype(np.float32),
        dtype=flow.int64,
        device=flow.device(device_type),
    )
    for i in range(10):
        activation = (np.random.random(activation_shape) - 0.5).astype(
            type_name_to_np_type[dtype]
        )
        activation_tensor = flow.tensor(
            activation, dtype=flow.float32, device=flow.device(device_type)
        )
        moving_average_min_max_observer = flow.nn.MovingAverageMinMaxObserver(
            stop_update_after_iters=1,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            momentum=momentum,
        )
        moving_average_min_max_observer = moving_average_min_max_observer.to(
            device_type
        )
        (scale, zero_point) = moving_average_min_max_observer(
            activation_tensor, current_train_step_tensor
        )
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


class TestMovingAverageMinMaxObserver(flow.unittest.TestCase):
    def test_moving_average_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["activation_shape"] = [(9, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["momentum"] = [0.95]
        for arg in GenArgList(arg_dict):
            _run_test_moving_average_min_max_observer(*arg)


if __name__ == "__main__":
    unittest.main()
