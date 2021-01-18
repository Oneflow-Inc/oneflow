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
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as tp 


def _random_input( x_shape):
    x = np.random.standard_normal(x_shape).astype(np.float32)
    # y = np.random.standard_normal(y_shape).astype(np.float32)
    return x

def diag_forward_np(input_tensor, dim):
    input_shape = input_tensor.shape
    input_dtype = input_tensor.dtype
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
                output_arr[i][int(beg%stride0)] = input_tensor[i]
            if dim < 0:
                print('beg%stride0 is {}'.format(beg%stride0))
                output_arr[int((beg - i)/stride0)][i] = input_tensor[i]

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
            output_arr[i] = input_tensor[i][int(beg/stride0)]
            #output_arr[i] = 1
        
        return output_arr

def diag_grad_np(input_tensor, dim, output, grad):
    print('output_tensor is {}'.format(output_tensor))
    input_shape = input_tensor.shape
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



def forward_cpmputer_with_np(device_type, input_tensor, dim):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.scope.placement("cpu", '0:0'))

    @flow.global_function(type="predict", function_config=func_config)
    def diag_forward(x: tp.Numpy.Placeholder((2,2)))-> tp.Numpy:
        y = flow.diag(x, 0)
        print('------------y---------------')
        print(y)
        return y

    #input = np.random.rand(*input_shape).astype(type_name_to_flow_type[dtype])
    #print(input)
    check_point = flow.train.CheckPoint()
    check_point.init()
    forward_of_out = diag_forward(input_tensor)
    print(forward_of_out)
   
    forward_np_out = diag_forward_np(input_tensor, dim)
    print(forward_np_out)
    assert np.allclose(forward_of_out, forward_np_out)


def backward_cpmputer_with_np(device_type, input_tensor, dim):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.scope.placement("cpu", '0:0'))


    output_tensor = diag_forward_np(input_tensor, dim)
    print('output_tensor shape is {}'.format(output_tensor.shape))
    output_shape = output_tensor.shape
    output_dtype = output_tensor.dtype
    grad = np.random.random(output_shape).astype(output_dtype)

    @flow.global_function(type="train", function_config=func_config)
    def DiagForwardJob():
            x = flow.get_variable(
                "input",
                shape=(2, 2),
                dtype=type_name_to_flow_type["float32"],
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )

            output = flow.diag(x, 0)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(output)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(output, test_global_storage.Setter("output"))
            flow.watch_diff(output, test_global_storage.Setter("output_diff"))

            return output

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    backward_of_out = DiagForwardJob(input_tensor).get()
    print(backward_of_out)

    backward_np_out = diag_grad_np(input_tensor, dim, output_tensor, grad)
    assert np.allclose(backward_of_out, backward_np_out)


def test_fun(device_type, input_shape, dim, dtype):
    input_tensor = np.random.random(input_shape).astype(dtype)
    input_tensor = x = input_tensor.reshape((2, 2)).astype(np.float32)
    print('input_tensor is {}'.format(input_tensor.shape))
    forward_cpmputer_with_np(device_type, input_tensor, dim)
    backward_cpmputer_with_np(device_type, input_tensor, dim)


@flow.unittest.skip_unless_1n1d()
class TestCast(flow.unittest.TestCase):
    def test_cast_forward(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["input_shape"] = [(2, 2)]
        arg_dict["dim"] = [0]
        arg_dict["dtype"] = ["float32"]
        for arg in GenArgList(arg_dict):
            test_fun( *arg)

if __name__ == "__main__":
    unittest.main()
    
