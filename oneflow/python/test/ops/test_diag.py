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

import unittest
import numpy as np
import oneflow as flow
import oneflow.typing as oft
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

def _random_input( x_shape):
    x = np.random.standard_normal(x_shape).astype(np.float32)
    # y = np.random.standard_normal(y_shape).astype(np.float32)
    return x

def diag_forward_compute(device_type, input_shape, dtype):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="predict", function_config=func_config)
    def diag_forward(
            input_def: oft.Numpy.Placeholder(
                shape=input_shape, dtype=type_name_to_flow_type[dtype]
            )
        ):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "input_tensor",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=False,
            )
        x = x + input_def
        y = flow.diag(x)
        return y

    #input = np.random.rand(*input_shape).astype(type_name_to_flow_type[dtype])
    #print(input)
    of_out = diag_forward()
    print(of_out)
    return
 
    '''
    @flow.unittest.skip_unless_1n1d()
    class TestCast(flow.unittest.TestCase):
        def test_cast_forward(test_case):
            arg_dict = OrderedDict()
            arg_dict["device_type"] = ["gpu"]
            arg_dict["input_shape"] = [(5, 4)]
            arg_dict["dtype"] = ["float32"]
            for arg in GenArgList(arg_dict):
                diag_forward_compute(test_case, *arg)
    
    if __name__ == "__main__":
        unittest.main()
    '''
if __name__ == '__main__':
    x_shape = (5, 4)
    x = _random_input(x_shape)
    device_type = "cpu"
    dtype = "float32"
    print(x)
    diag_forward_compute(device_type, x_shape, dtype)

    