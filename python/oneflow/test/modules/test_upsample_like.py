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
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


def _test_upsample_nearest_2d_like(test_case, shape_scale):
    input_shape, out_like_shape = shape_scale
    # init data by shape
    inputs = np.random.randn(*input_shape)
    out_like = np.random.randn(*out_like_shape)

    # get numpy function
    def nearest_upsample_by_np(inputs, out_like):
        in_height, in_width = inputs.shape[-2:]
        out_height, out_width = out_like.shape[-2:]
        scale_h = out_height / in_height
        scale_w = out_width / in_width
        output = np.zeros(out_like.shape)
        for i in range(out_height):
            for j in range(out_width):
                src_i = int(min(i / scale_h, in_height - 1))
                src_j = int(min(j / scale_w, in_width - 1))
                output[..., i, j] = inputs[..., src_i, src_j]
        return output

    # oneflow
    cpu_input = flow.tensor(inputs, dtype=flow.float32)
    cpu_out_like = flow.tensor(out_like, dtype=flow.float32)
    cpu_output = flow.nn.functional.interpolate_like(
        cpu_input, like=cpu_out_like, mode="nearest"
    )
    # numpy
    np_output = nearest_upsample_by_np(inputs, out_like)
    # compare result between oneflow and numpy
    test_case.assertTrue(np.allclose(np_output, cpu_output.numpy(), 0.001, 0.001))


@flow.unittest.skip_unless_1n1d()
class TestUpsample2dLike(flow.unittest.TestCase):
    def test_upsample2d_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_upsample_nearest_2d_like,
        ]
        arg_dict["shape_scale"] = [
            ((1, 1, 2, 2), (1, 1, 3, 3)),
            ((5, 3, 6, 4), (5, 3, 9, 6)),
            ((2, 3, 2, 4), (2, 3, 3, 5)),
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
