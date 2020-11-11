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


def _make_op_function(
    test_case,
    input,
    padding,
    data_format,
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
        test_case.assertTrue(np.allclose(grad, blob))

    if value_type == flow.float32 or value_type == flow.float64:

        @flow.global_function(type="train", function_config=func_config)
        def op_function(x: tp.Numpy.Placeholder(input.shape, dtype=value_type)):
            with flow.scope.placement(device_type, "0:0"):
                x += flow.get_variable(
                    name="v1",
                    shape=(1,),
                    dtype=value_type,
                    initializer=flow.zeros_initializer(),
                )
                out = flow.reflection_pad2d(x, padding, data_format)
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
                    name="v1",
                    shape=(1,),
                    dtype=flow.float32,
                    initializer=flow.zeros_initializer(),
                )
                y_int32 = flow.reflection_pad2d(x, padding, data_format)
                y_fp32 = flow.cast(y_int32, dtype=flow.float32)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
                ).minimize(y_fp32)

                flow.watch_diff(x, _compare_diff)
            return y_fp32

        return op_function


def gen_numpy_test_sample(input_shape, padding, data_format, is_float=True):
    def _flatten_array(input_array):
        output_array = list()
        for x in np.nditer(input_array):
            output_array.append(x.tolist())
        return output_array

    def _index2coordinate(idx, tensor_shape):
        coordinate = []
        tmp = idx
        for i in range(len(tensor_shape) - 1, -1, -1):
            axis_size = tensor_shape[i]
            coor = tmp % axis_size
            coordinate.insert(0, int(coor))
            tmp = (tmp - coor) / axis_size
        return coordinate

    def _coordinate2index(coordinate, tensor_shape):
        if len(coordinate) != len(tensor_shape):
            raise "wrong coordinate or shape"
        idx = 0
        for i, coor in enumerate(coordinate):
            size_at_axis = coor
            for j in range(i + 1, len(tensor_shape)):
                size_at_axis *= tensor_shape[j]

            idx += size_at_axis
        return idx

    def _np_reflection_pad2d(input, padding, data_format):
        if data_format == "NCHW":
            pad_top = pad_bottom = padding[2]
            pad_left = pad_right = padding[3]
            pad_shape = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        elif data_format == "NHWC":
            pad_top = pad_bottom = padding[1]
            pad_left = pad_right = padding[2]
            pad_shape = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            raise "data_format must be 'NCHW' or 'NHWC'"

        numpy_reflect = np.pad(input, pad_shape, "reflect")
        return numpy_reflect

    def _np_reflection_pad2d_grad(input, padding, data_format):
        numpy_reflect = np.ones(input.shape, np.int32)
        return numpy_reflect

    if is_float:
        input = np.random.random(input_shape).astype(np.float32)
    else:
        input = np.random.randint(0, 100, input_shape)
    output = _np_reflection_pad2d(input, padding, data_format)
    grad = _np_reflection_pad2d_grad(input, padding, data_format)

    numpy_results = {
        "input": input,
        "padding": padding,
        "data_format": data_format,
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
        sample["data_format"],
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
    arg_dict["samples"].append(
        gen_numpy_test_sample((1, 1, 2, 2), [0, 0, 1, 1], "NCHW")
    )
    arg_dict["samples"].append(
        gen_numpy_test_sample((3, 3, 3, 2), [0, 0, 2, 1], "NHWC")
    )
    arg_dict["samples"].append(
        gen_numpy_test_sample((2, 3, 4, 5), [0, 0, 2, 2], "NCHW")
    )
    if value_type == "float":
        arg_dict["value_type"] = [
            # (np.float32, flow.float16), #TODO:(ZhaoLuyang) float16 only works fine on ARCH > 700 CUDA > 10000
            (np.float32, flow.float32),
            (np.float64, flow.float64),
        ]
    elif value_type == "int":
        arg_dict["value_type"] = [(np.float32, flow.int32)]
    else:
        raise "float or int for value type only"

    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_count"] = [device_count]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestReflectionPad2d1n1d(flow.unittest.TestCase):
    def test_op_function_float_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    def test_op_function_int_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_float_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_op_function_int_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestReflectionPad2d1n2d(flow.unittest.TestCase):
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
