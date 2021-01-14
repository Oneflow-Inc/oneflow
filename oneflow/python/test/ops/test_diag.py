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

def diag_forward_np(input, dim):
    input_shape = input.shape
    input_dtype = input.dtype
    if len(input_shape) == 1:
        output_size = input_shape[0] + abs(dim)
        output_shape = [output_size, output_size]
        output_arr = np.zeros(output_shape)
        stride0 = output_size
        stride1 = 1

        beg = stride1*dim if dim >=0 else stride0 * abs(dim)
        for i in range(input_shape[0]):
            beg = beg + i * (stride1 + stride0)
            if dim >= 0:
                output_arr[i][int(beg%stride0)] = input[i]
            if dim < 0:
                print('beg%stride0 is {}'.format(beg%stride0))
                output_arr[int((beg - i)/stride0)][i] = input[i]

        return output_arr

    else:
        stride1 = 1
        stride0 = input_shape[1]
        beg = stride1*dim if dim <0 else stride0 * abs(dim)

        if dim >=0 :
            output_size = min(input_shape[0], input_shape[1] - dim)
        else:
            output_size = min(input_shape[0] + dim, input_shape[1])

        output_arr = np.zeros([output_size], dtype = input_dtype)
        for i in range(output_size):
            beg = beg + i * (stride1 + stride0)
            output_arr[i] = input[i][int(beg/stride0)]

        return output_arr

def diag_grad_np(input_ten, dim, output, grad):
    input_shape = input_ten.shape
    output_shape = output.shape
    grad_output = np.zeros(input_shape) 
    
    
    if len(input_shape) == 1:
        stride1 = 1
        stride0 = output_shape[1]
        beg = stride1*dim if dim >=0 else stride0 * abs(dim)
        for i in range(input_shape[0]):
            beg = beg + i * (stride1 + stride0)
            if dim >= 0:
                grad_output[i] = grad[i][int(beg%stride0)]
            if dim < 0:
                grad_output[i] = grad[int((beg - i)/stride0)][i]

        return grad_output    
    else:
        stride1 = 1
        stride01 = input_shape[1]
        beg = stride1*dim if dim >= 0 else stride01 * abs(dim)
        print(grad, grad_output, output_shape[0], beg)
        for i in range(output.shape[0]):
            beg = beg + i * (stride1 + stride01)
            if dim >= 0:
                print(beg%stride01)
                grad_output[i][int(beg%stride01)] = grad[i]
            if dim < 0:
                stride02 = input_shape[0]
                grad_output[int(beg%stride02)][i] = grad[i]
            
        return grad_output





def diag_forward_compute(device_type, input_shape, dtype):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="predict", function_config=func_config)
    def diag_forward():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "input_tensor",
                shape=[5, 4],
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=False,
            )
        x = x 
        y = flow.diag(x)
        return 

    #input = np.random.rand(*input_shape).astype(type_name_to_flow_type[dtype])
    #print(input)
    of_out = diag_forward()
    print(of_out)
    return of_out

def compare_oneflow_and_np(device_type, input_shape, dim):
    #oneflow
    diag_forward_of = diag_forward_compute(device_type, input_shape, dim)

    diag_backward_of = diag_forward_compute(device_type, input_shape, dim)

    #np
    diag_forward_np = diag_forward_np(device_type, input_shape, dim)

    diag_backward_np = diag_grad_np(device_type, input_shape, dim)

    assert np.array_equal(diag_forward_of, diag_forward_np) and assert np.array_equal(diag_backward_of, diag_backward_np)


 

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
    
