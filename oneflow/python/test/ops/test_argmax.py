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
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, in_shape, axis, data_type):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ArgMaxJob(
        input: oft.ListNumpy.Placeholder(
            tuple([dim + 10 for dim in in_shape]),
            dtype=type_name_to_flow_type[data_type],
        )
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.argmax(input, axis)

    input = (np.random.random(in_shape) * 100).astype(type_name_to_np_type[data_type])
    # OneFlow
    of_out = ArgMaxJob([input]).get().numpy_list()[0]
    # TensorFlow
    tf_out = tf.math.argmax(input, axis).numpy()
    tf_out = np.array([tf_out]) if isinstance(tf_out, np.int64) else tf_out

    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["in_shape"] = [
        (100,),
        (10, 10, 20),
        (10, 1000),
    ]
    arg_dict["axis"] = [-1]
    arg_dict["data_type"] = ["double", "int64"]

    return GenArgList(arg_dict)


def gen_arg_list_for_test_axis():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["in_shape"] = [(10, 10, 20, 30)]
    arg_dict["axis"] = [-2, 0, 1, 2]
    arg_dict["data_type"] = ["float32", "int32"]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestArgmax(flow.unittest.TestCase):
    def test_argmax(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_argmax_gpu(test_case):
        for arg in gen_arg_list_for_test_axis():
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
