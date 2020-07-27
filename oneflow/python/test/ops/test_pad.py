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
from test_util import Args, CompareOpWithTensorFlow, GenArgDict


def test_pad_gpu(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["flow_op"] = [flow.pad]
    arg_dict["tf_op"] = [tf.pad]
    arg_dict["input_shape"] = [(2, 2, 1, 3), (1, 1, 2, 3)]
    arg_dict["op_args"] = [
        Args(
            [([0, 0], [0, 0], [1, 2], [1, 1])],
            tf.constant([([0, 0], [0, 0], [1, 2], [1, 1])]),
        ),
        Args(
            [([0, 0], [0, 0], [0, 1], [1, 0])],
            tf.constant([([0, 0], [0, 0], [0, 1], [1, 0])]),
        ),
        Args(
            [([0, 0], [0, 0], [10, 20], [0, 0])],
            tf.constant([([0, 0], [0, 0], [10, 20], [0, 0])]),
        ),
    ]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)


def test_pad_cpu(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu"]
    arg_dict["flow_op"] = [flow.pad]
    arg_dict["tf_op"] = [tf.pad]
    arg_dict["input_shape"] = [(2, 3, 4, 3), (5, 1, 1, 1)]
    arg_dict["op_args"] = [
        Args(
            [([0, 0], [0, 0], [1, 2], [1, 1])],
            tf.constant([([0, 0], [0, 0], [1, 2], [1, 1])]),
        ),
        Args(
            [([0, 0], [0, 0], [0, 1], [1, 0])],
            tf.constant([([0, 0], [0, 0], [0, 1], [1, 0])]),
        ),
        Args(
            [([0, 0], [0, 0], [10, 20], [0, 0])],
            tf.constant([([0, 0], [0, 0], [10, 20], [0, 0])]),
        ),
    ]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)
