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

import os
import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, x_shape, axis):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def check_grad(x_diff_blob):
        assert np.array_equal(x_diff_blob.numpy(), np.ones(x_shape))

    @flow.global_function(type="train", function_config=func_config)
    def SqueezeJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "var",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.ones_initializer(),
                trainable=True,
            )
            flow.watch_diff(x, check_grad)
            loss = flow.squeeze(x, axis)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            return loss

    # OneFlow
    of_out = SqueezeJob().get().numpy()
    # TensorFlow
    tf_out = tf.squeeze(np.ones(x_shape, dtype=np.float32), axis).numpy()
    tf_out = np.array([tf_out]) if isinstance(tf_out, np.float32) else tf_out

    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(1, 10, 1, 10, 1)]
    arg_dict["axis"] = [None, [2], [-3], [0, 2, 4], [-1, -3, -5]]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestSqueeze(flow.unittest.TestCase):
    def test_squeeze(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)
        if os.getenv("ONEFLOW_TEST_CPU_ONLY") is None:
            compare_with_tensorflow("gpu", (1, 1, 1), [0, 1, 2])
            compare_with_tensorflow("gpu", (5, 6, 7), None)
        compare_with_tensorflow("cpu", (1, 1, 1), [0, 1, 2])
        compare_with_tensorflow("cpu", (5, 6, 7), None)


if __name__ == "__main__":
    unittest.main()
