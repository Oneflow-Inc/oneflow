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
from test_util import GenArgList

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def compare_with_tensorflow(test_case, device_type, value, shape, rtol=1e-5, atol=1e-5):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    @flow.global_function(function_config=func_config)
    def ConstantJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.constant(value, dtype=flow.float, shape=shape)
            y = flow.math.relu(x)
            z = flow.math.relu(y)
            return x

    numpy0 = ConstantJob().get().numpy()
    of_out = ConstantJob().get()
    test_case.assertTrue(np.allclose(of_out.numpy(), numpy0, rtol=rtol, atol=atol))
    tf_out = tf.constant(value, dtype=float, shape=shape)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)
    )


@flow.unittest.skip_unless_1n1d()
class TestConstant(flow.unittest.TestCase):
    def test_constant(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["value"] = [6, 6.66]
        arg_dict["shape"] = [(2, 3), (3, 3, 3)]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
