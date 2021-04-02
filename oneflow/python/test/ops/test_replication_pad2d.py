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
from test_util import (
    Args,
    GenArgDict,
    GenArgList,
    FlattenArray,
    Array2Numpy,
    Index2Coordinate,
    Coordinate2Index,
)


def _make_op_function(
    test_case,
    input,
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
        test_case.assertTrue(np.allclose(grad, blob, 1e-3, 1e-3))

    if value_type == flow.float32 or value_type == flow.float64:

        @flow.global_function(type="train", function_config=func_config)
        def op_function(x: tp.Numpy.Placeholder(input.shape, dtype=value_type)):
            with flow.scope.placement(device_type, "0:0"):
                x += flow.get_variable(
                    name="input",
                    shape=input.shape,
                    dtype=value_type,
                    initializer=flow.zeros_initializer(),
                )
                out = flow.replication_pad2d(x, padding)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
                ).minimize(out)

            flow.watch_diff(x, _compare_diff)
            return out

        return op_function

    elif value_type == flow.int32:

        @flow.global_function(type="train", function_config=func_config)
        def op_function(x: tp.Numpy.Placeholder(input.shape, dtype=flow.float32)):
            with flow.scope.placement(device_type, "0:0"):
                x += flow.get_variable(
                    name="input",
                    shape=input.shape,
                    dtype=flow.float32,
                    initializer=flow.zeros_initializer(),
                )
                y_int32 = flow.replication_pad2d(x, padding)
                y_fp32 = flow.cast(y_int32, dtype=flow.float32)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
                ).minimize(y_fp32)

            flow.watch_diff(x, _compare_diff)
            return y_fp32

        return op_function

    elif value_type == flow.float16:

        @flow.global_function(type="train", function_config=func_config)
        def op_function(x: tp.Numpy.Placeholder(input.shape, dtype=flow.float32)):
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    name="input",
                    shape=input.shape,
                    dtype=flow.float32,
                    initializer=flow.constant_initializer(0),
                )
                x_var = flow.cast_to_current_logical_view(x_var)
                input_x = x_var + x
                x_fp32 = flow.cast(input_x, flow.float32)
                x_fp16 = flow.cast(input_x, dtype=flow.float16)
                y_fp16 = flow.replication_pad2d(x_fp16, padding)
                y_fp32 = flow.cast(y_fp16, dtype=flow.float32)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
                ).minimize(y_fp32)

            flow.watch_diff(x_fp32, _compare_diff)
            return y_fp32

        return op_function


def gen_numpy_test_sample(input_shape, padding, is_float=True):
    c_idx, h_idx, w_idx = 1, 2, 3
    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    pad_shape = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))

    def _np_replication_pad2d(input, pad_shape):
        numpy_replicate = np.pad(input, pad_shape, "edge")
        return numpy_replicate

    def _np_replication_pad2d_grad(src, dest):
        dx_height, dx_width = input.shape[h_idx], input.shape[w_idx]
        dy_height, dy_width = output.shape[h_idx], output.shape[w_idx]

        numpy_src = np.ones(src.shape, np.int32)
        numpy_dest = np.zeros(dest.shape, np.int32)
        array_src = FlattenArray(numpy_src)
        array_dest = FlattenArray(numpy_dest)

        src_num = src.shape[c_idx] * src.shape[h_idx] * src.shape[w_idx]
        dest_num = dest.shape[c_idx] * dest.shape[h_idx] * dest.shape[w_idx]
        elements_num = src.shape[0] * src_num
        for iter_n in range(elements_num):
            coords = Index2Coordinate(iter_n, src.shape)
            n, c, i, j = coords[0], coords[c_idx], coords[h_idx], coords[w_idx]
            ip_x = ip_y = 0
            if j < pad_left:
                ip_x = pad_left
            elif j >= pad_left and j < (dx_width + pad_left):
                ip_x = j
            else:
                ip_x = dx_width + pad_left - 1

            if i < pad_top:
                ip_y = pad_top
            elif i >= pad_top and i < (dx_height + pad_top):
                ip_y = i
            else:
                ip_y = dx_height + pad_top - 1

            ip_x = ip_x - pad_left
            ip_y = ip_y - pad_top
            src_index = n * src_num + c * dy_width * dy_height + i * dy_width + j
            dest_index = (
                n * dest_num + c * dx_width * dx_height + ip_y * dx_width + ip_x
            )
            array_dest[dest_index] += array_src[src_index]

        numpy_dest = Array2Numpy(array_dest, dest.shape)
        return numpy_dest

    if is_float:
        input = np.random.random(input_shape).astype(np.float32)
    else:
        input = np.random.randint(0, 100, input_shape)

    output = _np_replication_pad2d(input, pad_shape)
    grad = _np_replication_pad2d_grad(output, input)

    numpy_results = {
        "input": input,
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
        sample["padding"],
        sample["grad"].astype(value_type[0]),
        device_type,
        value_type[1],
        machine_ids,
        device_count,
    )
    y = (
        op_function(sample["input"].astype(value_type[0]))
        .get()
        .numpy()
        .astype(value_type[0])
    )

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
    arg_dict["samples"].append(gen_numpy_test_sample((2, 1, 2, 2), [1, 1, 1, 1]))
    arg_dict["samples"].append(gen_numpy_test_sample((4, 2, 3, 3), [2, 2, 2, 2]))
    arg_dict["samples"].append(gen_numpy_test_sample((2, 3, 4, 5), [3, 2, 1, 2]))
    if value_type == "float":
        if device_type == "gpu":
            arg_dict["value_type"] = [
                (np.float32, flow.float32),
                # (np.float16, flow.float16),
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
class TestReplicationPad2d1n1d(flow.unittest.TestCase):
    def test_op_function_int_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    def test_op_function_float_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_int_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestReplicationPad2d1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_int(test_case):
        arg_dict = _gen_arg_dict("gpu", "int", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
