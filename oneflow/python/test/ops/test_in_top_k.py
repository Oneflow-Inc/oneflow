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
from collections import OrderedDict

import os
import unittest
import numpy as np
import oneflow as flow
import oneflow.typing as tp
import tensorflow as tf
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(
    device_type, target_dtype, predictions_shape, k, with_finite=False
):
    assert device_type in ["gpu", "cpu"]
    assert target_dtype in ["int32", "int64"]

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    instance_num = predictions_shape[0]
    classes = predictions_shape[1]
    targets = np.random.randint(classes, size=instance_num).astype(
        type_name_to_np_type[target_dtype]
    )
    predictions = np.random.rand(*predictions_shape).astype("float32")
    if with_finite:
        predictions[np.random.randint(instance_num)][
            np.random.randint(classes)
        ] = float("inf")

    @flow.global_function(function_config=func_config)
    def IntopkJob(
        targets: tp.ListNumpy.Placeholder(
            (instance_num + 10,), dtype=type_name_to_flow_type[target_dtype]
        ),
        predictions: tp.ListNumpy.Placeholder(
            tuple([dim + 5 for dim in predictions_shape]), dtype=flow.float
        ),
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.in_top_k(targets, predictions, k=k)

    # OneFlow
    of_out = IntopkJob([targets], [predictions]).get().numpy_list()[0]
    # TensorFlow
    tf_out = tf.math.in_top_k(targets, predictions, k=k)
    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["target_dtype"] = ["int32", "int64"]
    arg_dict["predictions_shape"] = [(10, 5)]
    arg_dict["k"] = [1, 2, 5]
    arg_dict["with_finite"] = [False, True]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestInTopk(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_in_top_K(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
