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
import oneflow.typing as oft
import test_global_storage
import random
import math
import unittest
import os

from test_util import GenArgList, type_name_to_flow_type
from collections import OrderedDict


def split_like(input, like, name):
    return (
        flow.user_op_builder(name)
        .Op("split_like")
        .Input("in", [input])
        .Input("like", like)
        .Output("out", len(like))
        .Attr("axis", 0)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )


def compare_with_np(device_type, x_shape, like0_shape, like1_shape, dtype):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def SplitLikeJob(x: oft.Numpy.Placeholder(x_shape, dtype=flow.float)):
        v = flow.get_variable(
            "x",
            shape=x_shape,
            dtype=flow.float,
            initializer=flow.constant_initializer(0),
            trainable=True,
        )
        x += v

        like0 = flow.constant(0, dtype=flow.float, shape=like0_shape)
        like1 = flow.constant(0, dtype=flow.float, shape=like1_shape)

        with flow.scope.placement("gpu", "0:0"):
            y0, y1 = split_like(x, [like0, like1], "split_like")
            loss = y0
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
        ).minimize(loss)

        flow.watch(x, test_global_storage.Setter("x"))
        flow.watch_diff(x, test_global_storage.Setter("x_diff"))
        flow.watch(loss, test_global_storage.Setter("loss"))
        flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

        return y0, y1

    # OneFlow
    x = np.random.randn(*x_shape).astype(np.float32)
    y0, y1 = SplitLikeJob(x).get()
    assert (like0_shape[0] + like1_shape[0]) == x_shape[0]
    np_y0 = x[0 : like0_shape[0]]
    np_y1 = x[like0_shape[0] :]
    zeros = np.zeros(np_y1.shape, dtype=np.float32)
    np_x_diff = np.concatenate([test_global_storage.Get("loss_diff"), zeros], axis=0)
    assert np.array_equal(y0.numpy(), np_y0)
    assert np.array_equal(y1.numpy(), np_y1)
    assert np.array_equal(test_global_storage.Get("x_diff"), np_x_diff)


@flow.unittest.skip_unless_1n1d()
class TestSplitLike(flow.unittest.TestCase):
    def test_split_like_axis0(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [(15, 20, 10)]
        arg_dict["like0_shape"] = [(10,)]
        arg_dict["like1_shape"] = [(5,)]
        arg_dict["dtype"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
