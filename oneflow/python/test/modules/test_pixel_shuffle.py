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


def _np_pixel_shuffle_v2(input, h_factor, w_factor):
    _batch, _channel, _height, _width = input.shape
    assert (
        _channel % (h_factor * w_factor) == 0
    ), "The channels of input tensor must be divisible by (h_upscale_factor * w_upscale_factor)"
    _new_c = int(_channel / (h_factor * w_factor))

    out = np.reshape(input, [_batch, _new_c, h_factor * w_factor, _height, _width])
    out = np.reshape(out, [_batch, _new_c, h_factor, w_factor, _height, _width])
    out = np.transpose(out, [0, 1, 4, 2, 5, 3])
    out = np.reshape(out, [_batch, _new_c, _height * h_factor, _width * w_factor])
    return out


def _np_pixel_shuffle_v2_grad(input, h_factor, w_factor):
    _batch, _new_channel, _height_mul_factor, _width_mul_factor = input.shape
    _channel = _new_channel * (h_factor * w_factor)
    _height = _height_mul_factor // h_factor
    _width = _width_mul_factor // w_factor

    out = np.ones(shape=(_batch, _channel, _height, _width))
    return out


if __name__ == "__main__":
    unittest.main()
