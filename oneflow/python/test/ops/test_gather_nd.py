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
import numpy as np
import unittest
from collections import OrderedDict
import oneflow as flow
from test_util import GenArgDict, type_name_to_flow_type, type_name_to_np_type


def _random_inputs(x_shape, x_dtype, index_shape, index_dtype):
    assert isinstance(x_shape, (tuple, list))
    assert isinstance(index_shape, (tuple, list))
    assert index_dtype == np.int32 or index_dtype == np.int64

    if x_dtype == np.float32 or x_dtype == np.double:
        x = np.random.rand(*x_shape).astype(x_dtype)
    elif x_dtype == np.int32 or x_dtype == np.int64 or x_dtype == np.int8:
        x = np.random.randint(low=0, high=100, size=x_shape).astype(x_dtype)
    else:
        raise NotImplementedError("{}".format(x_dtype))

    index = []
    index_rows = np.prod(index_shape[:-1])
    index_cols = index_shape[-1]
    for col in range(index_cols):
        index_col = np.random.randint(
            low=0, high=x_shape[col], size=(index_rows,), dtype=index_dtype
        ).reshape(index_shape[:-1])
        index.append(index_col)
    index = np.stack(index, axis=len(index_shape) - 1)
    return x, index


def _make_gather_nd_fn(
    x_shape,
    index_shape,
    x_dtype,
    index_type,
    device_type,
    device_num,
    dynamic,
    need_grad,
    comp_diff_fn,
):
    assert device_num >= 1
    fn_type = "train" if need_grad else "predict"

    if device_type == "gpu":
        flow.config.gpu_device_num(device_num)
    elif device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        raise ValueError

    func_config = flow.FunctionConfig()
    func_config.default_data_type(x_dtype)
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )
    if dynamic:
        func_config.default_logical_view(flow.scope.mirrored_view())
    else:
        func_config.default_logical_view(flow.scope.consistent_view())

    def do_gather_nd(x, index):
        x_var = flow.get_variable(
            "params", shape=(1,), dtype=x_dtype, initializer=flow.zeros_initializer(),
        )
        x = x + flow.cast_to_current_logical_view(x_var)
        y = flow.gather_nd(x, index)
        if need_grad:
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(y)
            if callable(comp_diff_fn):
                flow.watch_diff(x, comp_diff_fn)
        return y

    if dynamic:

        @flow.global_function(type=fn_type, function_config=func_config)
        def gather_nd_fn(
            x: flow.typing.Numpy.Placeholder(x_shape, dtype=x_dtype),
            index: flow.typing.Numpy.Placeholder(index_shape, dtype=index_type),
        ) -> flow.typing.Numpy:
            return do_gather_nd(x, index)

    else:

        @flow.global_function(type=fn_type, function_config=func_config)
        def gather_nd_fn(
            x: flow.typing.Numpy.Placeholder(x_shape, dtype=x_dtype),
            index: flow.typing.Numpy.Placeholder(index_shape, dtype=index_type),
        ) -> flow.typing.Numpy:
            return do_gather_nd(x, index)

    return gather_nd_fn


def _gather_nd_np(x, index, require_grad=False, init_grad_value=1.0):
    ndim = index.shape[-1]
    assert ndim <= x.ndim
    indices = []
    for dim in range(ndim):
        indices.append(index[..., dim])

    y = x[tuple(indices)]
    dy = None
    dx = None
    if require_grad:
        dy = np.zeros(shape=y.shape, dtype=np.float32)
        dy.fill(init_grad_value)
        dx = np.zeros(shape=x.shape, dtype=np.float32)
        flat_index = index.reshape(-1, ndim)
        flat_dy = dy.reshape(-1, *y.shape[(index.ndim - 1) :])
        for i, nd_index in enumerate(flat_index):
            if dx.ndim == ndim:
                ravel_index = np.ravel_multi_index(nd_index, dx.shape)
                dx_partial = np.zeros(shape=dx.shape, dtype=np.float32)
                np.put(dx_partial, ravel_index, flat_dy[i])
                dx += dx_partial
            else:
                dx[tuple(nd_index)] += flat_dy[i]

    return y, dx


def _is_floating_dtype(dtype):
    if dtype in ("float32", "double", "float16"):
        return True

    return False


def _compare_with_np(
    test_case,
    shape,
    index_shape,
    dynamic_shape=None,
    dynamic_index_shape=None,
    dtype="float32",
    index_dtype="int32",
    device_type="gpu",
    device_num=1,
    dynamic=False,
):
    x_is_floating = _is_floating_dtype(dtype)
    need_grad = True if x_is_floating else False
    x_of_dtype = type_name_to_flow_type[dtype]
    index_of_dtype = type_name_to_flow_type[index_dtype]
    x_dtype = type_name_to_np_type[dtype]
    index_dtype = type_name_to_np_type[index_dtype]

    if dynamic_shape is None:
        dynamic_shape = shape
    else:
        dynamic = True

    if dynamic_index_shape is None:
        dynamic_index_shape = index_shape
    else:
        dynamic = True

    if dynamic:
        x, index, y, dx = [], [], [], []
        for _ in range(device_num):
            x_, index_ = _random_inputs(
                dynamic_shape, x_dtype, dynamic_index_shape, index_dtype
            )
            y_, dx_ = _gather_nd_np(x_, index_, need_grad)
            x.append(x_)
            index.append(index_)
            y.append(y_)
            dx.append(dx_)

        def comp_diff(dx_blob: flow.typing.Numpy):
            for dx_blob_, dx_ in zip(dx_blob, dx):
                test_case.assertTrue(np.array_equal(dx_blob_, dx_))

    else:
        x, index = _random_inputs(
            dynamic_shape, x_dtype, dynamic_index_shape, index_dtype
        )
        y, dx = _gather_nd_np(x, index, need_grad)

        def comp_diff(dx_blob: flow.typing.Numpy):
            test_case.assertTrue(np.array_equal(dx_blob, dx))

    flow.clear_default_session()
    gather_nd_fn = _make_gather_nd_fn(
        shape,
        index_shape,
        x_of_dtype,
        index_of_dtype,
        device_type,
        device_num,
        dynamic,
        need_grad,
        comp_diff if device_num == 1 else None,
    )
    ret_y = gather_nd_fn(x, index)

    if dynamic:
        for ret_y_, y_ in zip(ret_y, y):
            test_case.assertTrue(np.array_equal(ret_y_, y_))
    else:
        test_case.assertTrue(np.array_equal(ret_y, y))


@flow.unittest.skip_unless_1n1d()
class TestGatherNd(flow.unittest.TestCase):
    def test_gather_nd(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10,)]
        arg_dict["index_shape"] = [(5, 1)]
        arg_dict["dtype"] = ["float32", "int32", "double"]
        arg_dict["index_dtype"] = ["int32", "int64"]
        arg_dict["device_type"] = ["gpu", "cpu"]
        # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
        # arg_dict["dynamic"] = [False, True]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_gather_nd_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(20, 10, 10, 3, 3)]
        arg_dict["index_shape"] = [(2, 3, 3)]
        arg_dict["device_type"] = ["gpu"]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_gather_nd_case_2(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10, 8, 4)]
        arg_dict["index_shape"] = [(2, 2)]
        arg_dict["dtype"] = ["float32", "int32"]
        arg_dict["index_dtype"] = ["int32", "int64"]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_gather_nd_case_3(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(32, 60, 80, 25)]
        arg_dict["index_shape"] = [(128, 2)]
        arg_dict["device_type"] = ["gpu"]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_gather_nd_case_4(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(128, 64, 2, 16, 7)]
        arg_dict["index_shape"] = [(30, 10, 3)]
        arg_dict["device_type"] = ["gpu"]
        arg_dict["dynamic"] = [True]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_with_dynamic_x(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(32, 16)]
        arg_dict["dynamic_shape"] = [(30, 15)]
        arg_dict["index_shape"] = [(12, 1)]
        arg_dict["device_type"] = ["cpu", "gpu"]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_with_dynamic_index(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(25, 10)]
        arg_dict["index_shape"] = [(15, 1)]
        arg_dict["dynamic_index_shape"] = [(11, 1)]
        arg_dict["device_type"] = ["cpu", "gpu"]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_with_empty_index(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(12, 13, 7)]
        arg_dict["index_shape"] = [(5, 10, 2)]
        arg_dict["dynamic_index_shape"] = [(5, 0, 2)]
        arg_dict["device_type"] = ["cpu", "gpu"]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)


@flow.unittest.skip_unless_1n4d()
class TestGatherNdParallel(flow.unittest.TestCase):
    def test_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(12, 5)]
        arg_dict["index_shape"] = [(4, 8, 2)]
        arg_dict["dtype"] = ["float32", "int32", "double"]
        arg_dict["index_dtype"] = ["int32", "int64"]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["device_num"] = [4]
        # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
        # arg_dict["dynamic"] = [True, False]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
