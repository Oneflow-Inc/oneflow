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
import math
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgDict, RunOneflowOp

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def tf_gelu(x):
    inv_sqrt2 = math.sqrt(0.5)
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        y = 0.5 * x * (1 + tf.math.erf(inv_sqrt2 * x))
    x_diff = tape.gradient(y, x)
    return y.numpy(), x_diff.numpy()


@flow.unittest.skip_unless_1n1d()
class TestGelu(flow.unittest.TestCase):
    def test_gelu(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["flow_op"] = [flow.math.gelu]
        arg_dict["flow_args"] = [[]]
        arg_dict["x"] = [
            np.random.uniform(low=-100, high=100, size=(10, 20, 30, 40)).astype(
                np.float32
            )
        ]
        for arg in GenArgDict(arg_dict):
            of_y, of_x_diff = RunOneflowOp(**arg)
            tf_y, tf_x_diff = tf_gelu(arg["x"])

            assert np.allclose(of_y, tf_y, rtol=1e-5, atol=1e-5)
            assert np.allclose(of_x_diff, tf_x_diff, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
