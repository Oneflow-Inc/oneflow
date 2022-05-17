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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _np_pixel_shuffle(input, h_factor, w_factor):
    (_batch, _channel, _height, _width) = input.shape
    assert (
        _channel % (h_factor * w_factor) == 0
    ), "The channels of input tensor must be divisible by (h_upscale_factor * w_upscale_factor)"
    _new_c = int(_channel / (h_factor * w_factor))
    out = np.reshape(input, [_batch, _new_c, h_factor * w_factor, _height, _width])
    out = np.reshape(out, [_batch, _new_c, h_factor, w_factor, _height, _width])
    out = np.transpose(out, [0, 1, 4, 2, 5, 3])
    out = np.reshape(out, [_batch, _new_c, _height * h_factor, _width * w_factor])
    return out


def _np_pixel_shuffle_grad(input, h_factor, w_factor):
    (_batch, _new_channel, _height_mul_factor, _width_mul_factor) = input.shape
    _channel = _new_channel * (h_factor * w_factor)
    _height = _height_mul_factor // h_factor
    _width = _width_mul_factor // w_factor
    out = np.ones(shape=(_batch, _channel, _height, _width))
    return out


def _test_pixel_shuffle_impl(
    test_case, device, shape, h_upscale_factor, w_upscale_factor
):
    x = np.random.randn(*shape)
    input = flow.tensor(
        x, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    m = flow.nn.PixelShuffle(
        h_upscale_factor=h_upscale_factor, w_upscale_factor=w_upscale_factor
    )
    m = m.to(device)
    of_out = m(input)
    np_out = _np_pixel_shuffle(x, h_upscale_factor, w_upscale_factor)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = _np_pixel_shuffle_grad(np_out, h_upscale_factor, w_upscale_factor)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestPixelShuffleModule(flow.unittest.TestCase):
    def test_pixel_shuffle(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_pixel_shuffle_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 144, 5, 5), (11, 144, 1, 1)]
        arg_dict["h_upscale_factor"] = [2, 3, 4]
        arg_dict["w_upscale_factor"] = [2, 3, 4]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
        arg_dict["shape"] = [(8, 25, 18, 18), (1, 25, 2, 2)]
        arg_dict["h_upscale_factor"] = [5]
        arg_dict["w_upscale_factor"] = [5]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest()
    def test_pixel_shuffle_with_random_data(test_case):
        upscale_factor = random().to(int)
        num_channels = upscale_factor * upscale_factor * random().to(int)
        m = torch.nn.PixelShuffle(upscale_factor=upscale_factor)
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim1=num_channels).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()
