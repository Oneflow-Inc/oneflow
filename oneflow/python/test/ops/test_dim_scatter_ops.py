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


flow.config.enable_debug_mode(True)


def _bin_add(out_val, in_value):
    return out_val + in_value


def _bin_update(out_val, in_value):
    return in_value


def gen_scatter_like_test_sample(
    input_shape,
    index_shape,
    dim,
    like_shape,
    is_float=True,
    binop=_bin_add,
    inplace=True,
):
    def _np_dim_scatter_add_like(input, dim, index, like):
        if inplace:
            output = like.copy()
        else:
            output = np.zeros(like.shape)

        for inputidx in range(0, input.size):
            outcoord = np.unravel_index(inputidx, input.shape)
            outcoord = [*outcoord]
            outcoord[dim] = index[np.unravel_index(inputidx, index.shape)]
            output_offset = np.ravel_multi_index(outcoord, like_shape)
            output[np.unravel_index(output_offset, like_shape)] = binop(
                output[np.unravel_index(output_offset, like_shape)],
                input[np.unravel_index(inputidx, input.shape)],
            )

        return output

    if is_float:
        input = np.random.random(input_shape)
        like = np.random.random(like_shape)
    else:
        input = np.random.randint(0, 100, input_shape)
        like = np.random.randint(0, 100, like_shape)

    def _np_dim_gather(dim, input, index):
        output = np.zeros(index.shape)
        for idx in range(0, index.size):
            incoord = np.unravel_index(idx, index.shape)
            outcoord = [*incoord]
            incoord = [*incoord]
            incoord[dim] = index[np.unravel_index(idx, index.shape)]
            output[tuple(outcoord)] = input[tuple(incoord)]
        return output

    index = np.random.randint(0, like_shape[dim], index_shape)

    output = _np_dim_scatter_add_like(input, dim, index, like)
    grad = _np_dim_gather(dim, np.ones(output.shape), index)
    return {
        "input": input,
        "index": index,
        "like": like,
        "dim": dim,
        "output": output,
        "grad": grad,
    }


def _gen_arg_dict(
    device_type="gpu",
    value_type="float",
    machine_ids="0:0",
    device_count=1,
    binop=_bin_add,
    dim_scatter_op=flow.dim_scatter_add_like,
    inplace=True,
):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = [device_type]
    arg_dict["samples"] = []
    arg_dict["samples"].append(
        gen_scatter_like_test_sample(
            (2, 2),
            (2, 2),
            1,
            (2, 2),
            is_float=value_type == "float",
            binop=binop,
            inplace=inplace,
        )
    )
    # arg_dict["samples"].append(gen_scatter_like_test_sample((2, 2), (2, 2), 0, (4, 4), value_type=="float"))
    # arg_dict["samples"].append(gen_scatter_like_test_sample((4, 3, 3), (4, 3, 3), 0, (5, 5, 5), value_type=="float"))
    if value_type == "float":
        arg_dict["value_type"] = [
            (np.float32, flow.float32),
        ]
    elif value_type == "int":
        arg_dict["value_type"] = [(np.float32, flow.int32)]
    else:
        raise "float or int for value type only"

    arg_dict["index_type"] = [(np.int32, flow.int32)]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_count"] = [device_count]
    arg_dict["flow_scatter_op"] = [dim_scatter_op]
    return arg_dict


def _make_dim_scatter_add_like_fn(
    test_case,
    input,
    index,
    dim,
    like,
    grad,
    device_type,
    value_type,
    index_type,
    machine_ids,
    device_counts,
    flow_scatter_op,
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

    if value_type == flow.float32 or value_type == flow.float64:

        @flow.global_function(type="train", function_config=func_config)
        def scatter_add_like_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=value_type),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
            like_def: oft.Numpy.Placeholder(like.shape, dtype=value_type),
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

            y = flow_scatter_op(dim, indices_def, x, like_def)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(y)

            flow.watch_diff(x, _compare_diff)
            return y

        return scatter_add_like_fn

    if value_type == flow.int32:

        @flow.global_function(type="train", function_config=func_config)
        def scatter_add_like_fn(
            params_def: oft.Numpy.Placeholder(input.shape, dtype=flow.float32),
            indices_def: oft.Numpy.Placeholder(index.shape, dtype=index_type),
            like_def: oft.Numpy.Placeholder(like.shape, dtype=flow.float32),
        ) -> oft.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    "input",
                    shape=params_def.shape,
                    dtype=flow.float32,
                    initializer=flow.constant_initializer(0),
                )
                x_var = flow.cast_to_current_logical_view(x_var)
                x = x_var + params_def

            x_int32 = flow.cast(x, dtype=flow.int32)
            like_def_int32 = flow.cast(like_def, dtype=flow.int32)
            y_int32 = flow_scatter_op(dim, indices_def, x_int32, like_def_int32)
            y_fp32 = flow.cast(y_int32, dtype=flow.int32)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(y_fp32)

            flow.watch_diff(x, _compare_diff)
            return y_fp32

        return scatter_add_like_fn


def _compare_dim_scatter_op_like_with_samples(
    test_case,
    device_type,
    sample,
    value_type,
    index_type,
    machine_ids,
    device_count,
    flow_scatter_op,
):
    scatter_add_like_fn = _make_dim_scatter_add_like_fn(
        test_case,
        sample["input"].astype(value_type[0]),
        sample["index"].astype(index_type[0]),
        sample["dim"],
        sample["like"].astype(value_type[0]),
        sample["grad"].astype(value_type[0]),
        device_type,
        value_type[1],
        index_type[1],
        machine_ids,
        device_count,
        flow_scatter_op,
    )
    y = scatter_add_like_fn(
        sample["input"].astype(value_type[0]),
        sample["index"].astype(index_type[0]),
        sample["like"].astype(value_type[0]),
    )
    y.astype(value_type[0])

    if value_type == flow.float16:
        test_case.assertTrue(
            np.allclose(y, sample["output"].astype(np.float32), 1e-3, 1e-3)
        )
    else:
        test_case.assertTrue(np.allclose(y, sample["output"].astype(value_type[0])))


@flow.unittest.skip_unless_1n1d()
class TestDimScatterOpsLike1n1d(flow.unittest.TestCase):
    def test_dim_scatter_add_like_int_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu", "int", "0:0", 1, _bin_add, flow.dim_scatter_add_like, inplace=False
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    def test_dim_scatter_add_like_float_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu", "float", "0:0", 1, _bin_add, flow.dim_scatter_add_like, inplace=False
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_add_like_int_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "int", "0:0", 1, _bin_add, flow.dim_scatter_add_like, inplace=False
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_add_like_float_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "float", "0:0", 1, _bin_add, flow.dim_scatter_add_like, inplace=False
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    def test_dim_scatter_update_like_int_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu",
            "int",
            "0:0",
            1,
            _bin_update,
            flow.dim_scatter_update_like,
            inplace=False,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    def test_dim_scatter_update_like_float_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu",
            "float",
            "0:0",
            1,
            _bin_update,
            flow.dim_scatter_update_like,
            inplace=False,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_update_like_int_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu",
            "int",
            "0:0",
            1,
            _bin_update,
            flow.dim_scatter_update_like,
            inplace=False,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_update_like_float_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu",
            "float",
            "0:0",
            1,
            _bin_update,
            flow.dim_scatter_update_like,
            inplace=False,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestDimScatterOpsLike1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_add_like_float(test_case):
        arg_dict = _gen_arg_dict(
            "gpu",
            "float",
            "0:0-1",
            2,
            _bin_add,
            flow.dim_scatter_add_like,
            inplace=False,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_update_like_float(test_case):
        arg_dict = _gen_arg_dict(
            "gpu",
            "float",
            "0:0-1",
            2,
            _bin_update,
            flow.dim_scatter_update_like,
            inplace=False,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n1d()
class TestDimScatterOpsInplace1n1d(flow.unittest.TestCase):
    def test_dim_scatter_add_int_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu", "int", "0:0", 1, _bin_add, flow.dim_scatter_add, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    def test_dim_scatter_add_float_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu", "float", "0:0", 1, _bin_add, flow.dim_scatter_add, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_add_like_int_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "int", "0:0", 1, _bin_add, flow.dim_scatter_add, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_add_like_float_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "float", "0:0", 1, _bin_add, flow.dim_scatter_add, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    def test_dim_scatter_update_like_int_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu", "int", "0:0", 1, _bin_update, flow.dim_scatter_update, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    def test_dim_scatter_update_like_float_cpu(test_case):
        arg_dict = _gen_arg_dict(
            "cpu", "float", "0:0", 1, _bin_update, flow.dim_scatter_update, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_update_like_int_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "int", "0:0", 1, _bin_update, flow.dim_scatter_update, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_update_like_float_gpu(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "float", "0:0", 1, _bin_update, flow.dim_scatter_update, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestDimScatterOpsInplace1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_add_like_float(test_case):
        arg_dict = _gen_arg_dict(
            "gpu", "float", "0:0-1", 2, _bin_add, flow.dim_scatter_add, inplace=True
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_scatter_update_like_float(test_case):
        arg_dict = _gen_arg_dict(
            "gpu",
            "float",
            "0:0-1",
            2,
            _bin_update,
            flow.dim_scatter_update,
            inplace=True,
        )
        for arg in GenArgList(arg_dict):
            _compare_dim_scatter_op_like_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
