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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import CompareOpWithTensorFlow, GenArgDict

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@flow.unittest.skip_unless_1n1d()
class TestSqrt(flow.unittest.TestCase):
    def test_sqrt(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["flow_op"] = [flow.math.sqrt]
        arg_dict["tf_op"] = [tf.math.sqrt]
        arg_dict["input_shape"] = [(10, 20, 30)]
        arg_dict["input_minval"] = [0]
        arg_dict["input_maxval"] = [100]
        for arg in GenArgDict(arg_dict):
            CompareOpWithTensorFlow(**arg)


if __name__ == "__main__":
    unittest.main()
