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
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
import os


def _compare_pixel_shuffle_with_np(
    input_shape, upscale_factor, device_type, machine_ids, device_counts
):
    input_1 = np.random.random(size=input_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    def np_pixel_shuffle(input):

        _batch, _channel, _height, _width = input.shape
        assert (
            _channel % (upscale_factor ** 2) == 0
        ), "The channels of input tensor must be divisible by (upscale_factor * upscale_factor)"

        _new_c = int(_channel / (upscale_factor ** 2))

        out = np.reshape(
            input, [_batch, _new_c, upscale_factor * upscale_factor, _height, _width]
        )
        out = np.reshape(
            out, [_batch * _new_c, upscale_factor, upscale_factor, _height, _width]
        )
        out = np.transpose(out, [0, 3, 1, 4, 2])
        out = np.reshape(
            out, [_batch, _new_c, _height * upscale_factor, _width * upscale_factor]
        )

        return out

    np_out_pixel_shuffle = np_pixel_shuffle(input_1)

    np_random_mul = np.random.random(size=np_out_pixel_shuffle.shape).astype(np.float32)

    def np_pixel_shuffle_diff(input, np_diff, upscale_factor):
        _batch, _new_channel, _height_mul_factor, _width_mul_factor = input.shape
        _channel = _new_channel * (upscale_factor ** 2)
        _height = _height_mul_factor // upscale_factor
        _width = _width_mul_factor // upscale_factor

        bp_result = np.zeros((_batch, _channel, _height, _width)).astype(np.float32)
        for c in range(_channel):
            for h in range(_height):
                for w in range(_width):
                    out_c_idx = int(c / (upscale_factor ** 2))
                    inner_c = c - out_c_idx * upscale_factor * upscale_factor
                    out_h_idx = h * upscale_factor + int(inner_c / upscale_factor)
                    out_w_idx = w * upscale_factor + int(inner_c % upscale_factor)
                    bp_result[:, c, h, w] = np_diff[:, out_c_idx, out_h_idx, out_w_idx]

        return bp_result

    _np_grad = np_pixel_shuffle_diff(
        np_out_pixel_shuffle, np_random_mul, upscale_factor
    )

    def assert_prediction_grad(blob: tp.Numpy):
        assert np.allclose(blob, _np_grad)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_pixel_shuffle(
        of_input_1: tp.Numpy.Placeholder(shape=input_1.shape),
        of_mul: tp.Numpy.Placeholder(shape=np_random_mul.shape),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=input_1.shape,
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
                name="x_var",
            )
            x_var = of_input_1 + v

        flow.watch_diff(x_var, assert_prediction_grad)

        of_pixel_shuffle_out = flow.nn.PixelShuffle(
            x_var, upscale_factor, name="PixelShuffle"
        )

        out = of_pixel_shuffle_out * of_mul

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(out)

        return of_pixel_shuffle_out

    of_out_pixel_shuffle = oneflow_pixel_shuffle(input_1, np_random_mul)

    assert np.allclose(of_out_pixel_shuffle, np_out_pixel_shuffle)


def _gen_arg_dict(shape, upscale_factor, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["upscale_factor"] = [upscale_factor]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestPixelShuffle1n1d(flow.unittest.TestCase):
    def test_pixel_shuffle_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 4, 2, 2),
            upscale_factor=2,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_pixel_shuffle_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_pixel_shuffle_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 2, 2),
            upscale_factor=2,
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_pixel_shuffle_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestPixelShuffle1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_pixel_shuffle_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 16, 2, 2),
            upscale_factor=2,
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_pixel_shuffle_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
