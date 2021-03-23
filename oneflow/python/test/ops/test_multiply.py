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
from test_util import (
    Args,
    CompareOpWithTensorFlow,
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _test_element_wise_mul_fw_bw(test_case, device, shape, type_name):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    np_type = type_name_to_np_type[type_name]
    flow_type = type_name_to_flow_type[type_name]

    @flow.global_function(type="train", function_config=func_config)
    def test_element_wise_mul_job(
        x: oft.Numpy.Placeholder(shape, dtype=flow.float),
        y: oft.Numpy.Placeholder(shape, dtype=flow.float),
    ):
        with flow.scope.placement(device, "0:0"):
            x += flow.get_variable(
                name="vx",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            y += flow.get_variable(
                name="vy",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x = flow.cast(x, dtype=flow_type)
            y = flow.cast(y, dtype=flow_type)
            out = flow.math.multiply(x, y)
            out = flow.cast(out, dtype=flow.float)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(y, test_global_storage.Setter("y"))
            flow.watch_diff(y, test_global_storage.Setter("y_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))
            return out

    x = np.random.randint(low=0, high=10, size=shape).astype(np.float32)
    y = np.random.randint(low=0, high=10, size=shape).astype(np.float32)
    test_element_wise_mul_job(x, y).get()
    test_case.assertTrue(
        np.allclose(
            test_global_storage.Get("x") * test_global_storage.Get("y"),
            test_global_storage.Get("out"),
        )
    )
    test_case.assertTrue(
        np.allclose(
            test_global_storage.Get("out_diff") * test_global_storage.Get("x"),
            test_global_storage.Get("y_diff"),
        )
    )
    test_case.assertTrue(
        np.allclose(
            test_global_storage.Get("out_diff") * test_global_storage.Get("y"),
            test_global_storage.Get("x_diff"),
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestMultiply(flow.unittest.TestCase):
    def test_scalar_mul(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.multiply]
        arg_dict["tf_op"] = [tf.math.multiply]
        arg_dict["input_shape"] = [(10, 10, 10)]
        arg_dict["op_args"] = [
            Args([1]),
            Args([-1]),
            Args([84223.19348]),
            Args([-3284.139]),
        ]
        for arg in GenArgDict(arg_dict):
            CompareOpWithTensorFlow(**arg)

    def test_element_wise_mul_fw_bw(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["gpu", "cpu"]
        arg_dict["shape"] = [(96, 96)]
        arg_dict["type_name"] = ["float32", "double", "int8", "int32", "int64"]
        for arg in GenArgDict(arg_dict):
            _test_element_wise_mul_fw_bw(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
