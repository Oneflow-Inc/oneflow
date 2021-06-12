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

import oneflow.experimental as flow
from test_util import GenArgList


def _np_pixel_shuffle(input, factor):
    _batch, _channel, _height, _width = input.shape
    assert (
        _channel % (factor ** 2) == 0
    ), "The channels of input tensor must be divisible by (upscale_factor * upscale_factor)"
    _new_c = int(_channel / (factor ** 2))

    out = np.reshape(input, [_batch, _new_c, factor ** 2, _height, _width])
    out = np.reshape(out, [_batch, _new_c, factor, factor, _height, _width])
    out = np.transpose(out, [0, 1, 4, 2, 5, 3])
    out = np.reshape(out, [_batch, _new_c, _height * factor, _width * factor])
    return out


def _np_pixel_shuffle_grad(input, factor):
    _batch, _new_channel, _height_mul_factor, _width_mul_factor = input.shape
    _channel = _new_channel * (factor ** 2)
    _height = _height_mul_factor // factor
    _width = _width_mul_factor // factor

    out = np.ones(shape=(_batch, _channel, _height, _width))
    return out


def _test_pixel_shuffle_impl(test_case, device, shape, upscale_factor):
    x = np.random.randn(*shape)
    input = flow.Tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )

    m = flow.nn.PixelShuffle(upscale_factor)
    m = m.to(device)
    of_out = m(input)
    np_out = _np_pixel_shuffle(x, upscale_factor)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_pixel_shuffle_grad(np_out, upscale_factor)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPixelShuffleModule(flow.unittest.TestCase):
    def test_pixel_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_pixel_shuffle_impl,
        ]
        arg_dict["device"] = ["cpu", "cuda"]

        arg_dict["shape"] = [(2, 144, 5, 5), (11, 144, 1, 1)]
        arg_dict["upscale_factor"] = [2, 3, 4]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

        arg_dict["shape"] = [(8, 25, 18, 18), (1, 25, 2, 2)]
        arg_dict["upscale_factor"] = [5]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
