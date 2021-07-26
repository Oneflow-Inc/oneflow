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
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp


def getExpandGrad(input_shape, expand_size):
    input = np.random.random(size=input_shape).astype(np.float32)
    input_stride = [1]
    for i in range(len(input_shape) - 2, -1, -1):
        input_stride.insert(0, input_stride[0] * input_shape[i + 1])
    new_size = []
    new_stride = []
    diff = len(expand_size) - len(input_shape)
    for i in range(len(expand_size) - 1, -1, -1):
        if i >= diff:
            if expand_size[i] == -1 or expand_size[i] == input_shape[i - diff]:
                new_size.insert(0, input_shape[i - diff])
                new_stride.insert(0, input_stride[i - diff])
            else:
                assert expand_size[i] >= 1 and input_shape[i - diff] == 1
                new_size.insert(0, expand_size[i])
                new_stride.insert(0, 0)
        else:
            assert expand_size[i] >= 1
            new_size.insert(0, expand_size[i])
            if expand_size[i] == 1:
                new_stride.insert(0, new_stride[0])
            else:
                new_stride.insert(0, 0)
    gout = np.random.random(size=tuple(new_size)).astype(np.float32)
    out_stride = [1]
    for i in range(len(new_size) - 2, -1, -1):
        out_stride.insert(0, out_stride[0] * new_size[i + 1])
    gin = np.zeros(input_shape).flatten()
    out = np.zeros(np.product(new_size))

    def getOffset(i_offset, stride, expand_stride, n):
        remain = i_offset
        o_offset = 0
        for i in range(n):
            idx = int(remain / stride[i])
            o_offset += idx * expand_stride[i]
            remain = remain - idx * stride[i]
        return o_offset

    in_flatten = input.flatten()
    gout_flatten = gout.flatten()
    num_elem = np.product(new_size)
    dims = len(new_size)
    for i in range(num_elem):
        offset = getOffset(i, out_stride, new_stride, dims)
        gin[offset] += gout_flatten[i]
        out[i] = in_flatten[offset]
    return (input, gout, out.reshape(tuple(new_size)), gin.reshape(input_shape))


def _compare_expand_op_with_np(
    input_shape, expand_dim, data_type, device_type, machine_ids, device_counts
):
    assert device_type in ["cpu", "gpu"]
    if device_type == "cpu" and data_type == flow.float16:
        return
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)
    func_config = flow.FunctionConfig()
    if data_type == flow.float16:
        func_config.default_data_type(flow.float32)
    else:
        func_config.default_data_type(data_type)
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    (input, gout, out_np, gin_np) = getExpandGrad(input_shape, expand_dim)

    def assert_prediction_grad(gin_of: tp.Numpy):
        assert np.allclose(gin_of, gin_np, atol=1e-05)

    if data_type == flow.float32:

        @flow.global_function(type="train", function_config=func_config)
        def expandJob(
            of_input: tp.Numpy.Placeholder(shape=input.shape, dtype=data_type),
            multipler: tp.Numpy.Placeholder(shape=gout.shape, dtype=data_type),
        ) -> tp.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                v = flow.get_variable(
                    shape=of_input.shape,
                    dtype=data_type,
                    initializer=flow.constant_initializer(0),
                    name="v",
                )
                x_var = of_input + v
                flow.watch_diff(x_var, assert_prediction_grad)
            out = flow.expand(x_var, expand_dim)
            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
                ).minimize(out * multipler)
            return out

        of_out = expandJob(input, gout)
        assert np.allclose(of_out, out_np, atol=1e-05)
    elif data_type == flow.float64:

        @flow.global_function(type="train", function_config=func_config)
        def expandJob(
            of_input: tp.Numpy.Placeholder(shape=input.shape, dtype=flow.float32),
            multipler: tp.Numpy.Placeholder(
                shape=gout.shape, dtype=flow.float32, batch_axis=diff
            ),
        ) -> tp.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                v = flow.get_variable(
                    shape=of_input.shape,
                    dtype=flow.float32,
                    initializer=flow.constant_initializer(0),
                    name="v",
                )
                input_x = v + of_input
                flow.watch_diff(input_x, assert_prediction_grad)
            x_fp32 = flow.cast(input_x, flow.float32)
            x_fp16 = flow.cast(input_x, dtype=flow.float16)
            y_fp16 = flow.expand(x_fp16, expand_dim)
            y_fp32 = flow.cast(y_fp16, dtype=flow.float32)
            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
                ).minimize(y_fp32 * multipler)
            return y_fp32

        of_out = expandJob(input, gout)
        assert np.allclose(of_out, out_np, atol=1e-05)


@flow.unittest.skip_unless_1n1d()
class TestExpandOp1n1d(flow.unittest.TestCase):
    def test_expand(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_shape"] = [(1, 4, 1, 32)]
        arg_dict["expand_dim"] = [[1, 4, 2, 32]]
        arg_dict["expand_dim"] = [[2, 4, 2, 32], [2, 1, 2, 4, 2, 32]]
        arg_dict["data_type"] = [flow.float32, flow.float16]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_expand_op_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestExpandOp1n2d(flow.unittest.TestCase):
    def test_expand(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_shape"] = [(2, 4, 1, 32)]
        arg_dict["expand_dim"] = [[2, 4, 2, 32], [2, 1, 2, 4, 2, 32]]
        arg_dict["data_type"] = [flow.float32, flow.float16]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        for arg in GenArgList(arg_dict):
            _compare_expand_op_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
