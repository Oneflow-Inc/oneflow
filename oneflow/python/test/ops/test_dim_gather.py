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
import oneflow as flow
import numpy as np
import oneflow.typing as oft
from test_util import GenArgList
import unittest
from collections import OrderedDict
import os


def gen_gather_test_sample(input_shape, index_shape, dim, is_float=True):
    def _np_dim_scatter_add(src, dim, index, outshape):
        output = np.zeros(outshape)
        for srcidx in range(0, src.size):
            outcoord = np.unravel_index(srcidx, src.shape)
            outcoord = [*outcoord]
            outcoord[dim] = index[np.unravel_index(srcidx, index.shape)]
            output_offset = np.ravel_multi_index(outcoord, outshape)
            output[np.unravel_index(output_offset, outshape)] += src[
                np.unravel_index(srcidx, src.shape)
            ]
        return output

    if is_float:
        input = np.random.random(input_shape)
    else:
        input = np.random.randint(0, 100, input_shape)
    index = np.random.randint(0, input_shape[dim], index_shape)
    output = np.take_along_axis(input, index, dim)
    grad = _np_dim_scatter_add(np.ones_like(output), dim, index, input_shape)

    ret = {"input": input, "index": index, "dim": dim, "output": output, "grad": grad}
    return ret


def _make_dim_gather_fn(
    test_case,
    input,
    index,
    dim,
    grad,
    device_type,
    value_type,
    index_type,
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

    def _compare_diff(blob: oft.Numpy):
        test_case.assertTrue(np.allclose(grad, blob))

    if value_type == flow.float16:

        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=flow.float32),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
        ) -> oft.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=flow.float32,
                    initializer=flow.constant_initializer(0),
                )
                x_var = flow.cast_to_current_logical_view(x_var)
                x = x_var + params_def
                x_f16 = flow.cast(x, flow.float16)

            y_f16 = flow.dim_gather(x_f16, dim, indices_def)
            x_f32 = flow.cast(x, flow.float32)
            y_f32 = flow.cast(y_f16, flow.float32)

            y = flow.dim_gather(x, dim, indices_def)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(y_f32)

            flow.watch_diff(x_f32, _compare_diff)
            return y_f32

        return gather_fn
    elif value_type == flow.float32 or value_type == flow.float64:

        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=value_type),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
        ) -> oft.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=value_type,
                    initializer=flow.constant_initializer(0),
                )
                x_var = flow.cast_to_current_logical_view(x_var)
                x = x_var + params_def

            y = flow.dim_gather(x, dim, indices_def)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(y)

            flow.watch_diff(x, _compare_diff)
            return y

        return gather_fn
    elif value_type == flow.int32:

        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=flow.float32),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
        ) -> oft.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=flow.float32,
                    initializer=flow.constant_initializer(0),
                )
                x_var = flow.cast_to_current_logical_view(x_var)
                x = x_var + params_def

            x_int32 = flow.cast(x, dtype=flow.int32)
            y_int32 = flow.dim_gather(x, dim, indices_def)
            y_fp32 = flow.cast(y_int32, dtype=flow.float32)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(y_fp32)

            flow.watch_diff(x, _compare_diff)
            return y_fp32

        return gather_fn


def _compare_dim_gather_with_samples(
    test_case, device_type, sample, value_type, index_type, machine_ids, device_count
):
    gather_fn = _make_dim_gather_fn(
        test_case,
        sample["input"].astype(value_type[0]),
        sample["index"].astype(index_type[0]),
        sample["dim"],
        sample["grad"].astype(value_type[0]),
        device_type,
        value_type[1],
        index_type[1],
        machine_ids,
        device_count,
    )
    y = gather_fn(
        sample["input"].astype(value_type[0]), sample["index"].astype(index_type[0])
    )
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
    arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 1))
    arg_dict["samples"].append(gen_gather_test_sample((2, 2), (2, 2), 0))
    arg_dict["samples"].append(gen_gather_test_sample((8, 3, 2), (4, 3, 2), 0))
    if value_type == "float":
        arg_dict["value_type"] = [
            # (np.float32, flow.float16), #TODO:(YaoChi) float16 only works fine on ARCH > 700 CUDA > 10000
            (np.float32, flow.float32),
            (np.float64, flow.float64),
        ]
    elif value_type == "int":
        arg_dict["value_type"] = [(np.float32, flow.int32)]
    else:
        raise "float or int for value type only"

    arg_dict["index_type"] = [(np.int32, flow.int32), (np.int64, flow.int64)]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_count"] = [device_count]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestDimGather1n1d(flow.unittest.TestCase):
    def test_dim_gather_float_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

    def test_dim_gather_int_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_gather_float_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_gather_int_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "int", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestDimGather1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_gather_float(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_gather_int(test_case):
        arg_dict = _gen_arg_dict("gpu", "int", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
