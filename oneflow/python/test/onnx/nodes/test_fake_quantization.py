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


def generate_fake_quantization_test(
    per_layer: bool = True,
    formula: str = "google",
    scheme: str = "symmetric",
    device_type: str = "cpu",
):
    @flow.global_function()
    def fake_quantization():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                name="x1",
                shape=(2, 3, 4),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(-10, 10),
            )
            return flow.quantization.fake_quantization(
                x,
                *flow.quantization.min_max_observer(
                    x,
                    per_layer_quantization=per_layer,
                    quantization_formula=formula,
                    quantization_scheme=scheme,
                ),
                quantization_formula=formula,
                quantization_scheme=scheme,
            )

    convert_to_onnx_and_check(fake_quantization, opset=10 if per_layer else 13)


def generate_fake_quantization_test_moving_average(
    formula: str = "google", scheme: str = "symmetric", device_type: str = "cpu",
):
    @flow.global_function()
    def fake_quantization_moving_average():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                name="x1",
                shape=(2, 3, 4),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(-10, 10),
            )
            return flow.quantization.fake_quantization(
                x,
                *flow.quantization.moving_average_min_max_observer(
                    x, quantization_formula=formula, quantization_scheme=scheme,
                ),
                quantization_formula=formula,
                quantization_scheme=scheme,
            )

    set_moving_max_min_value()

    convert_to_onnx_and_check(
        fake_quantization_moving_average, opset=10, explicit_init=False
    )


# min_max_observer
def test_fake_quantization_symmetric(test_case):
    generate_fake_quantization_test(
        per_layer=True, formula="google", scheme="symmetric"
    )


def test_fake_quantization_symmetric_per_channel(test_case):
    generate_fake_quantization_test(
        per_layer=False, formula="google", scheme="symmetric"
    )


def test_fake_quantization_affine(test_case):
    generate_fake_quantization_test(per_layer=True, formula="google", scheme="affine")


def test_fake_quantization_affine_per_channel(test_case):
    generate_fake_quantization_test(per_layer=False, formula="google", scheme="affine")


def test_fake_quantization_cambricon(test_case):
    generate_fake_quantization_test(per_layer=True, formula="cambricon")


def test_fake_quantization_symmetric_gpu(test_case):
    generate_fake_quantization_test(
        per_layer=True, formula="google", scheme="symmetric", device_type="gpu"
    )


def test_fake_quantization_symmetric_per_channel_gpu(test_case):
    generate_fake_quantization_test(
        per_layer=False, formula="google", scheme="symmetric", device_type="gpu"
    )


def test_fake_quantization_affine_gpu(test_case):
    generate_fake_quantization_test(
        per_layer=True, formula="google", scheme="affine", device_type="gpu"
    )


def test_fake_quantization_affine_per_channel_gpu(test_case):
    generate_fake_quantization_test(
        per_layer=False, formula="google", scheme="affine", device_type="gpu"
    )


def test_fake_quantization_cambricon_gpu(test_case):
    generate_fake_quantization_test(
        per_layer=True, formula="cambricon", device_type="gpu"
    )


# moving_average_min_max_observer
def test_fake_quantization_symmetric_moving_average(test_case):
    generate_fake_quantization_test_moving_average(formula="google", scheme="symmetric")


def test_fake_quantization_affine_moving_average(test_case):
    generate_fake_quantization_test_moving_average(formula="google", scheme="affine")


def test_fake_quantization_cambricon_moving_average(test_case):
    generate_fake_quantization_test_moving_average(formula="cambricon")


def test_fake_quantization_symmetric_gpu_moving_average(test_case):
    generate_fake_quantization_test_moving_average(
        formula="google", scheme="symmetric", device_type="gpu"
    )


def test_fake_quantization_affine_gpu_moving_average(test_case):
    generate_fake_quantization_test_moving_average(
        formula="google", scheme="affine", device_type="gpu"
    )


def test_fake_quantization_cambricon_gpu_moving_average(test_case):
    generate_fake_quantization_test_moving_average(
        formula="cambricon", device_type="gpu"
    )
