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


def compare_with_tensorflow(device_type, in_shape, k, data_type, sorted):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def TopKJob(
        input: oft.ListNumpy.Placeholder(
            tuple([dim + 10 for dim in in_shape]),
            dtype=type_name_to_flow_type[data_type],
        )
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.top_k(input, k, sorted)

    input = (np.random.random(in_shape) * 100).astype(type_name_to_np_type[data_type])
    # OneFlow
    of_out = TopKJob([input]).get().numpy_list()[0]
    # TensorFlow
    if k <= in_shape[-1]:
        _, tf_out = tf.math.top_k(input, k, sorted)
    else:
        tf_out = tf.argsort(input, axis=-1, direction="DESCENDING", stable=True)

    assert np.array_equal(of_out, tf_out.numpy())


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(100,), (100, 100), (10, 500), (10, 10, 500)]
    arg_dict["k"] = [1, 50, 200]
    arg_dict["data_type"] = ["float32", "double", "int32", "int64"]
    arg_dict["sorted"] = [True]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestTopK(flow.unittest.TestCase):
    def test_top_k(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
