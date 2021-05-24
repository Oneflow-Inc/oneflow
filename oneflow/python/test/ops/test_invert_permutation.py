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

import numpy as np
import oneflow as flow
import oneflow.typing as tp
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import random


def compare_with_tensorflow(device_type, in_shape, data_type):
    assert device_type in ["cpu", "gpu"]
    assert data_type in ["int32", "int64"]
    assert len(in_shape) == 1
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.int32)

    @flow.global_function(function_config=func_config)
    def InvertJob(
        x: tp.Numpy.Placeholder(shape=in_shape, dtype=type_name_to_flow_type[data_type])
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.invert_permutation(x)

    x = np.array(random.sample(range(0, in_shape[0]), in_shape[0])).astype(
        type_name_to_np_type[data_type]
    )
    # OneFlow
    of_out = InvertJob(x).get().numpy()
    # TensorFlow
    tf_out = tf.math.invert_permutation(x).numpy()
    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(5,), (10,)]
    arg_dict["data_type"] = ["int32", "int64"]

    return GenArgList(arg_dict)


def test_invertpermutation(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
