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
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _check(test_case, x, y, depth, on_value, off_value, axis):
    out = tf.one_hot(x, depth=depth, axis=axis, on_value=on_value, off_value=off_value)
    test_case.assertTrue(np.array_equal(out.numpy(), y))


def _run_test(
    test_case, device_type, x_shape, depth, dtype, out_dtype, on_value, off_value, axis
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def one_hot_job(
        x: oft.Numpy.Placeholder(x_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.one_hot(
                x,
                depth=depth,
                on_value=on_value,
                off_value=off_value,
                axis=axis,
                dtype=type_name_to_flow_type[out_dtype],
            )

    x = np.random.randint(0, depth, x_shape).astype(type_name_to_np_type[dtype])
    y = one_hot_job(x).get()
    _check(test_case, x, y.numpy(), depth, on_value, off_value, axis)


@flow.unittest.skip_unless_1n1d()
class TestOneHot(flow.unittest.TestCase):
    def test_one_hot(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [(10, 20, 20)]
        arg_dict["depth"] = [10]
        arg_dict["dtype"] = ["int32", "int64"]
        arg_dict["out_dtype"] = ["int32", "double"]
        arg_dict["on_value"] = [5]
        arg_dict["off_value"] = [2]
        arg_dict["axis"] = [-1, 0, 2]
        for arg in GenArgList(arg_dict):
            _run_test(*arg)


if __name__ == "__main__":
    unittest.main()
