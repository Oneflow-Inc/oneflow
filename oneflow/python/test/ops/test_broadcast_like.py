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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList
import oneflow.typing as oft


def compare_broadcast_like_with_tf(
    device_type, input_shape, like_shape, broadcast_axes, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def broadcast_like_forward(
        x: oft.Numpy.Placeholder(shape=input_shape, dtype=flow.float),
        y: oft.Numpy.Placeholder(shape=like_shape, dtype=flow.float),
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.broadcast_like(x, y, broadcast_axes=broadcast_axes)

    x = np.random.rand(*input_shape).astype(np.float32)
    like = np.random.rand(*like_shape).astype(np.float32)
    of_out = broadcast_like_forward(x, like).get()
    np_out = np.broadcast_to(x, like_shape)
    assert np.allclose(of_out.numpy(), np_out, rtol=rtol, atol=atol)


def test_broadcast_like(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(5, 2)]
    arg_dict["like_shape"] = [(4, 5, 2)]
    arg_dict["broadcast_axes"] = [[0]]
    for arg in GenArgList(arg_dict):
        compare_broadcast_like_with_tf(*arg)


def test_broadcast_like2(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(5, 2)]
    arg_dict["like_shape"] = [(4, 6, 5, 2)]
    arg_dict["broadcast_axes"] = [[0, 1]]
    for arg in GenArgList(arg_dict):
        compare_broadcast_like_with_tf(*arg)
