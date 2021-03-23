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
import numpy as np
import oneflow as flow
from util import convert_to_onnx_and_check


def set_moving_max_min_value():
    max_key, min_key = "", ""
    keys = flow.get_all_variables().keys()
    for key in keys:
        if max_key != "" and min_key != "":
            break
        if key[-3:] == "max":
            max_key = key
        if key[-3:] == "min":
            min_key = key
    flow.load_variables(
        {
            max_key: np.array([0.5]).astype(np.float32),
            min_key: np.array([-0.2]).astype(np.float32),
        }
    )


def generate_moving_average_min_max_observer_test(
    out_pos: int, formula: str, scheme: str = "symmetric", device_type: str = "cpu",
):
    flow.clear_default_session()

    @flow.global_function()
    def moving_average_min_max_observer():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                name="x1",
                shape=(2, 3, 4),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(-10, 10),
            )
            return flow.quantization.moving_average_min_max_observer(
                x, quantization_formula=formula, quantization_scheme=scheme
            )[out_pos]

    set_moving_max_min_value()

    convert_to_onnx_and_check(
        moving_average_min_max_observer, opset=10, explicit_init=False
    )


def test_moving_average_min_max_observer_symmetric(test_case):
    generate_moving_average_min_max_observer_test(0, "google", "symmetric")


def test_moving_average_min_max_observer_symmetric_zero_point(test_case):
    generate_moving_average_min_max_observer_test(1, "google", "symmetric")


def test_moving_average_min_max_observer_affine(test_case):
    generate_moving_average_min_max_observer_test(0, "google", "affine")


def test_moving_average_min_max_observer_affine_zero_point(test_case):
    generate_moving_average_min_max_observer_test(1, "google", "affine")


def test_moving_average_min_max_observer_cambricon(test_case):
    generate_moving_average_min_max_observer_test(0, "cambricon")


def test_moving_average_min_max_observer_cambricon_zero_point(test_case):
    generate_moving_average_min_max_observer_test(1, "cambricon")


def test_moving_average_min_max_observer_symmetric_gpu(test_case):
    generate_moving_average_min_max_observer_test(
        0, "google", "symmetric", device_type="gpu"
    )


def test_moving_average_min_max_observer_symmetric_zero_point_gpu(test_case):
    generate_moving_average_min_max_observer_test(
        1, "google", "symmetric", device_type="gpu"
    )


def test_moving_average_min_max_observer_affine_gpu(test_case):
    generate_moving_average_min_max_observer_test(
        0, "google", "affine", device_type="gpu"
    )


def test_moving_average_min_max_observer_affine_zero_point_gpu(test_case):
    generate_moving_average_min_max_observer_test(
        1, "google", "affine", device_type="gpu"
    )


def test_moving_average_min_max_observer_cambricon_gpu(test_case):
    generate_moving_average_min_max_observer_test(0, "cambricon", device_type="gpu")


def test_moving_average_min_max_observer_cambricon_zero_point_gpu(test_case):
    generate_moving_average_min_max_observer_test(1, "cambricon", device_type="gpu")
