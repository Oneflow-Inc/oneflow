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
import oneflow as flow
from util import convert_to_onnx_and_check


def generate_min_max_observer_test(
    out_pos: int,
    per_layer: bool,
    formula: str,
    scheme: str,
    device_type: str = "cpu",
    opset: int = 10,
):
    @flow.global_function()
    def min_max_observer():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                name="x1",
                shape=(2, 3, 4),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(-10, 10),
            )
            return flow.quantization.min_max_observer(
                x,
                per_layer_quantization=per_layer,
                quantization_formula=formula,
                quantization_scheme=scheme,
            )[out_pos]

    convert_to_onnx_and_check(min_max_observer, opset=opset)


def test_min_max_observer_symmetric(test_case):
    generate_min_max_observer_test(0, True, "google", "symmetric")


def test_min_max_observer_symmetric_zero_point(test_case):
    generate_min_max_observer_test(1, True, "google", "symmetric")


def test_min_max_observer_affine(test_case):
    generate_min_max_observer_test(0, True, "google", "affine")


def test_min_max_observer_affine_zero_point(test_case):
    generate_min_max_observer_test(1, True, "google", "affine")


def test_min_max_observer_symmetric_per_channel(test_case):
    generate_min_max_observer_test(0, False, "google", "symmetric", opset=13)


def test_min_max_observer_symmetric_per_channel_zero_point(test_case):
    generate_min_max_observer_test(1, False, "google", "symmetric", opset=13)


def test_min_max_observer_affine_per_channel(test_case):
    generate_min_max_observer_test(0, False, "google", "affine", opset=13)


def test_min_max_observer_affine_per_channel_zero_point(test_case):
    generate_min_max_observer_test(1, False, "google", "affine", opset=13)


def test_min_max_observer_cambricon(test_case):
    generate_min_max_observer_test(0, True, "cambricon", "symmetric")


def test_min_max_observer_cambricon_zero_point(test_case):
    generate_min_max_observer_test(1, True, "cambricon", "symmetric")


def test_min_max_observer_symmetric_gpu(test_case):
    generate_min_max_observer_test(0, True, "google", "symmetric", device_type="gpu")


def test_min_max_observer_symmetric_zero_point_gpu(test_case):
    generate_min_max_observer_test(1, True, "google", "symmetric", device_type="gpu")


def test_min_max_observer_affine_gpu(test_case):
    generate_min_max_observer_test(0, True, "google", "affine", device_type="gpu")


def test_min_max_observer_affine_zero_point_gpu(test_case):
    generate_min_max_observer_test(1, True, "google", "affine", device_type="gpu")


def test_min_max_observer_symmetric_per_channel_gpu(test_case):
    generate_min_max_observer_test(
        0, False, "google", "symmetric", device_type="gpu", opset=13
    )


def test_min_max_observer_symmetric_per_channel_zero_point_gpu(test_case):
    generate_min_max_observer_test(
        1, False, "google", "symmetric", device_type="gpu", opset=13
    )


def test_min_max_observer_affine_per_channel_gpu(test_case):
    generate_min_max_observer_test(
        0, False, "google", "affine", device_type="gpu", opset=13
    )


def test_min_max_observer_affine_per_channel_zero_point_gpu(test_case):
    generate_min_max_observer_test(
        1, False, "google", "affine", device_type="gpu", opset=13
    )


def test_min_max_observer_cambricon_gpu(test_case):
    generate_min_max_observer_test(0, True, "cambricon", "symmetric", device_type="gpu")


def test_min_max_observer_cambricon_zero_point_gpu(test_case):
    generate_min_max_observer_test(1, True, "cambricon", "symmetric", device_type="gpu")
