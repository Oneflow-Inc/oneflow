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
import oneflow.typing as tp
from test_util import Args, GenArgDict, GenArgList


class MaxPool2D:
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)):
        self.stride = stride
        self.padding = padding

        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]

        self.x = None
        self.pad_x = None

        self.in_batch = None
        self.in_channel = None
        self.in_height = None
        self.in_width = None
        self.pad_shape = None

        self.out_height = None
        self.out_width = None

        self.arg_max = None

    def __call__(self, x):
        self.x = x
        self.in_batch = np.shape(x)[0]
        self.in_channel = np.shape(x)[1]
        self.in_height = np.shape(x)[2]
        self.in_width = np.shape(x)[3]

        pad_x = np.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                       'constant', constant_values=(0, 0))
        self.pad_x = pad_x
        self.pad_shape = pad_x.shape

        self.out_height = int((self.in_height - self.w_height) / self.stride[0]) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride[1]) + 1
        self.pad_out_height = np.uint16(round((self.pad_shape[2] - self.w_height + 1) / self.stride[0]))
        self.pad_out_width = np.uint16(round((self.pad_shape[3] - self.w_width + 1) / self.stride[1]))

        out = np.zeros((self.in_batch, self.in_channel, self.pad_out_height, self.pad_out_width))
        self.arg_max = np.zeros_like(out, dtype=np.int32)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_height):
                    for j in range(self.pad_out_width):
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]
                        end_i = start_i + self.w_height
                        end_j = start_j + self.w_width
                        out[n, c, i, j] = np.max(pad_x[n, c, start_i: end_i, start_j: end_j])
                        self.arg_max[n, c, i, j] = np.argmax(pad_x[n, c, start_i: end_i, start_j: end_j])

        self.arg_max = self.arg_max
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.pad_x)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_height):
                    for j in range(self.pad_out_width):
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]
                        end_i = start_i + self.w_height
                        end_j = start_j + self.w_width
                        index = np.unravel_index(self.arg_max[n, c, i, j], self.kernel_size)
                        dx[n, c, start_i: end_i, start_j: end_j][index] = d_loss[n, c, i, j]
        dx = dx[:,:,self.padding[0]:self.pad_shape[2]-self.padding[0],self.padding[1]:self.pad_shape[3]-self.padding[1]]
        return dx


def flatten_array(input_array):
    output_array = list()
    for x in np.nditer(input_array):
        output_array.append(x.tolist())
    return output_array


def array_to_numpy(input_array, target_shape):
    return np.array(input_array).reshape(target_shape, order="C")


def index2coordinate(idx, tensor_shape):
    coordinate = []
    tmp = idx
    for i in range(len(tensor_shape) - 1, -1, -1):
        axis_size = tensor_shape[i]
        coor = tmp % axis_size
        coordinate.insert(0, int(coor))
        tmp = (tmp - coor) / axis_size
    return coordinate


def coordinate2index(coordinate, tensor_shape):
    if len(coordinate) != len(tensor_shape):
        raise "wrong coordinate or shape"
    idx = 0
    for i, coor in enumerate(coordinate):
        size_at_axis = coor
        for j in range(i + 1, len(tensor_shape)):
            size_at_axis *= tensor_shape[j]

        idx += size_at_axis
    return idx


def _make_op_function(
    test_case,
    input,
    kernel_size,
    stride,
    padding,
    grad,
    device_type,
    value_type,
    machine_ids,
    device_counts,
):
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    # global function needs float32 as type of argument and return value
    if value_type == flow.float16:
        func_config.default_data_type(flow.float32)
    else:
        func_config.default_data_type(value_type)

    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    def _compare_diff(blob: tp.Numpy):
        flag = np.allclose(grad, blob, 1e-3, 1e-3)
        test_case.assertTrue(flag)

    if value_type == flow.float32 or value_type == flow.float64:

        @flow.global_function(type="train", function_config=func_config)
        def op_function(
            input_x: tp.Numpy.Placeholder(input.shape, dtype=value_type)
            ) -> tp.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    name="input",
                    shape=input.shape,
                    dtype=value_type,
                    initializer=flow.zeros_initializer(),
                    trainable=True
                )
                x_var = flow.cast_to_current_logical_view(x_var)
                x = x_var + input_x
            y = flow.nn.MaxPool2d(
                x,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=False,
                ceil_mode=False,
                data_format="NCHW",
            )
            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
                ).minimize(y)
          
            flow.watch_diff(x, _compare_diff)
            return y

        return op_function

    elif value_type == flow.int32:
        @flow.global_function(type="train", function_config=func_config)
        def op_function(
            input_x: tp.Numpy.Placeholder(input.shape, dtype=flow.float32)
            ) -> tp.Numpy:
            x_var = flow.get_variable(
                name="input_x",
                shape=input.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                trainable=True
            )
            x_var = flow.cast_to_current_logical_view(x_var)
            flow.watch_diff(x_var, _compare_diff)
            x = x_var + input_x
            x_int32 = flow.cast(x, dtype=flow.int32)
            with flow.scope.placement(device_type, "0:0"):
                y = flow.nn.MaxPool2d(
                    x_int32,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    return_indices=False,
                    ceil_mode=False,
                    data_format="NCHW",
                )
                y_fp32 = flow.cast(y, dtype=flow.float32)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
                ).minimize(y_fp32)
            return y_fp32

        return op_function


def gen_numpy_test_sample(input_shape, kernel_size, stride, padding, is_float=True):
    max_pool_numpy = MaxPool2D(kernel_size, stride, padding)


    def _np_maxpool2d(input):
        y = max_pool_numpy(input)
        return y

    def _np_maxpool2d_grad(input):
        grad = max_pool_numpy.backward(np.ones(input.shape))
        return grad

    if is_float:
        elm_cnt = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
        input = np.arange(elm_cnt).reshape(input_shape).astype(np.float32)
    else:
        input = np.random.randint(0, 100, input_shape)

    output = _np_maxpool2d(input)
    grad = _np_maxpool2d_grad(input)

    numpy_results = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output": output,
        "grad": grad,
    }

    return numpy_results


def _compare_op_function_with_samples(
    test_case, device_type, sample, value_type, machine_ids, device_count
):
    op_function = _make_op_function(
        test_case,
        sample["input"].astype(value_type[0]),
        sample["kernel_size"],
        sample["stride"],
        sample["padding"],
        sample["grad"].astype(value_type[0]),
        device_type,
        value_type[1],
        machine_ids,
        device_count,
    )
    y = op_function(sample["input"].astype(value_type[0]))
    y.astype(value_type[0])

    if value_type == flow.float16:
        test_case.assertTrue(
            np.allclose(y, sample["output"].astype(np.float32), 1e-3, 1e-3)
        )
    else:
        test_case.assertTrue(np.allclose(y, sample["output"].astype(value_type[0])))


def _gen_arg_dict(
    device_type="gpu", value_type="float", machine_ids="0:0", device_count=1
):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = [device_type]
    arg_dict["samples"] = []
    arg_dict["samples"].append(gen_numpy_test_sample((1, 2, 4, 4), (3,3), (1,1), (0,0)))
    arg_dict["samples"].append(gen_numpy_test_sample((1, 2, 6, 6), (3,3), (2,2), (1,1)))
    if value_type == "float":
        if device_type == "gpu":
            arg_dict["value_type"] = [
                (np.float32, flow.float32),
            ]
        else:
            arg_dict["value_type"] = [(np.float32, flow.float32)]

    elif value_type == "int":
        arg_dict["value_type"] = [(np.float32, flow.int32)]
    else:
        raise "float or int for value type only"

    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_count"] = [device_count]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestMaxPool2d1n1d(flow.unittest.TestCase):
    def test_op_function_int_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    def test_op_function_float_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)
    
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float_gpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestMaxPool2d1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float(test_case):
        arg_dict = _gen_arg_dict("cpu", "float", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
