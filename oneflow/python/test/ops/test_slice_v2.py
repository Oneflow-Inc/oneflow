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
import numpy as np
import oneflow as flow
import oneflow.typing as otp
import test_util
import typing as tp
import collections
import unittest
import os

DEFAULT_DEVICE_TAG = "gpu"
if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
    DEFAULT_DEVICE_TAG = "cpu"


def _do_slice(input, args, name=None):
    outputs = []
    for slice_tup_list in args:
        output = flow.slice_v2(input, slice_tup_list, name)
        outputs.append(output)
    return outputs


def _make_slice_func(slice_args, input_shape, dtype=flow.float32, func_cfg=None):
    @flow.global_function(type="predict", function_config=func_cfg)
    def slice_job(
        x: otp.Numpy.Placeholder(shape=input_shape, dtype=dtype)
    ) -> tp.List[otp.Numpy]:
        return _do_slice(x, slice_args)

    return slice_job


def _make_slice_with_fp16_func(slice_args, input_shape, func_cfg=None):
    @flow.global_function(type="predict", function_config=func_cfg)
    def slice_job(
        x: otp.Numpy.Placeholder(shape=input_shape, dtype=flow.float32)
    ) -> tp.List[otp.Numpy]:
        x = flow.cast(x, flow.float16)
        y = _do_slice(x, slice_args)
        return [flow.cast(y_i, flow.float32) for y_i in y]

    return slice_job


def _make_slice_dynamic_func(
    slice_args, input_shape, dtype=flow.float32, func_cfg=None
):
    if func_cfg is None:
        func_cfg = flow.FunctionConfig()
        func_cfg.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(type="predict", function_config=func_cfg)
    def slice_dynamic_job(
        x: otp.Numpy.Placeholder(shape=input_shape, dtype=dtype)
    ) -> tp.List[otp.Numpy]:
        return _do_slice(x, slice_args, name="SliceDynamic")

    return slice_dynamic_job


def _make_slice_with_grad_func(
    slice_tup_list, input_shape, watch_diff_cb=None, dtype=flow.float32, func_cfg=None,
):
    @flow.global_function(type="train", function_config=func_cfg)
    def slice_with_grad_job(
        x: otp.Numpy.Placeholder(shape=input_shape, dtype=dtype)
    ) -> otp.Numpy:
        var = flow.get_variable(
            shape=input_shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0.0),
            name="variable",
        )
        x = x + var
        if callable(watch_diff_cb):
            flow.watch_diff(x, watch_diff_cb)

        y = flow.slice_v2(x, slice_tup_list, name="SliceWithGrad")
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(y)
        return y

    return slice_with_grad_job


def _make_slice_update_func(
    slice_tup_list, input_shape, update_shape, dtype=flow.float32, func_cfg=None
):
    @flow.global_function(type="predict", function_config=func_cfg)
    def slice_update_job(
        x: otp.Numpy.Placeholder(shape=input_shape, dtype=dtype),
        update: otp.Numpy.Placeholder(shape=update_shape, dtype=dtype),
    ) -> otp.Numpy:
        return flow.slice_update(x, update, slice_tup_list)

    return slice_update_job


def _make_slice_update_grad_func(
    slice_tup_list,
    input_shape,
    update_shape,
    diff_watcher_maker=None,
    dtype=flow.float32,
    func_cfg=None,
):
    @flow.global_function(type="train", function_config=func_cfg)
    def slice_update_train_job(
        x: otp.Numpy.Placeholder(shape=input_shape, dtype=dtype),
        update: otp.Numpy.Placeholder(shape=update_shape, dtype=dtype),
    ) -> otp.Numpy:
        x_var = flow.get_variable(
            shape=input_shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0.0),
            name="x",
        )
        update_var = flow.get_variable(
            shape=update_shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0.0),
            name="update",
        )
        x = x + x_var
        update = update + update_var
        if callable(diff_watcher_maker):
            flow.watch_diff(x, diff_watcher_maker(input_shape))
            flow.watch_diff(update, diff_watcher_maker(update_shape))

        y = flow.slice_update(x, update, slice_tup_list)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(y)
        return y

    return slice_update_train_job


def _test_slice(
    test_case,
    input,
    slice_args,
    outputs,
    dtype=flow.float32,
    device_tag=DEFAULT_DEVICE_TAG,
    verbose=False,
):
    input = input.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    outputs = [
        output.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
        for output in outputs
    ]

    flow.clear_default_session()
    func_cfg = flow.FunctionConfig()
    func_cfg.default_data_type(dtype)
    func_cfg.default_placement_scope(flow.scope.placement(device_tag, "0:0"))
    slice_func = _make_slice_func(slice_args, input.shape, dtype, func_cfg)
    of_outputs = slice_func(input)

    if verbose:
        print("input:\n{}".format(input))
        print("slice_args:", slice_args)
        print("dtype:", dtype)
        print("device_tag:", device_tag)

    for out, of_out in zip(outputs, of_outputs):
        if verbose:
            print("output:\n{}\n{}".format(out, of_out))
        test_case.assertTrue(np.array_equal(out, of_out))


def _test_slice_dynamic(
    test_case,
    input,
    slice_args,
    outputs,
    static_shape=None,
    dtype=flow.float32,
    device_tag=DEFAULT_DEVICE_TAG,
):
    input = input.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    outputs = [
        output.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
        for output in outputs
    ]

    if static_shape is None:
        static_shape = input.shape

    flow.clear_default_session()
    func_cfg = flow.FunctionConfig()
    func_cfg.default_data_type(dtype)
    func_cfg.default_placement_scope(flow.scope.placement(device_tag, "0:0"))
    func_cfg.default_logical_view(flow.scope.mirrored_view())
    slice_func = _make_slice_dynamic_func(slice_args, static_shape, dtype, func_cfg)
    of_outputs = slice_func([input])
    for out, of_out in zip(outputs, of_outputs):
        test_case.assertTrue(np.array_equal(out, of_out[0]))


def _test_slice_with_grad(
    test_case,
    input,
    slice_args,
    output,
    diff,
    dtype=flow.float32,
    device_tag=DEFAULT_DEVICE_TAG,
    verbose=False,
):
    input = input.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    output = output.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    diff = diff.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    if verbose:
        print("dtype: {}".format(dtype))
        print("device_tag: {}".format(device_tag))
        print("input: {}\n{}\n".format(input.shape, input))
        print("output: {}\n{}\n".format(output.shape, output))
        print("diff: {}\n{}\n".format(diff.shape, diff))

    def WatchDiff(of_diff: otp.Numpy):
        if verbose:
            print("of_diff: {}\n{}\n".format(of_diff.shape, of_diff))
        test_case.assertTrue(np.array_equal(of_diff, diff))

    flow.clear_default_session()
    func_cfg = flow.FunctionConfig()
    func_cfg.default_data_type(dtype)
    func_cfg.default_placement_scope(flow.scope.placement(device_tag, "0:0"))
    slice_func = _make_slice_with_grad_func(
        slice_args, input.shape, WatchDiff, dtype, func_cfg
    )

    of_output = slice_func(input)
    if verbose:
        print("of_output: {}\n{}\n".format(of_output.shape, of_output))
    test_case.assertTrue(np.array_equal(output, of_output))


def _test_slice_update(
    test_case,
    input,
    update,
    slice_args,
    output,
    dtype=flow.float32,
    device_tag=DEFAULT_DEVICE_TAG,
    verbose=False,
):
    input = input.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    update = update.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    output = output.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))

    flow.clear_default_session()
    func_cfg = flow.FunctionConfig()
    func_cfg.default_data_type(dtype)
    func_cfg.default_placement_scope(flow.scope.placement(device_tag, "0:0"))
    slice_func = _make_slice_update_func(
        slice_args, input.shape, update.shape, dtype, func_cfg
    )
    of_output = slice_func(input, update)

    if verbose:
        print("input:\n{}".format(input))
        print("update:\n{}".format(update))
        print("slice_args:", slice_args)
        print("output:\n{}".format(output))
        print("dtype:", dtype)
        print("device_tag:", device_tag)
        print("of_output:\n{}".format(of_output))

    test_case.assertTrue(np.array_equal(output, of_output))


def _test_slice_update_grad(
    test_case,
    input,
    update,
    slice_args,
    output,
    input_diff,
    update_diff,
    dtype=flow.float32,
    device_tag=DEFAULT_DEVICE_TAG,
    verbose=False,
):
    input = input.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    update = update.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    output = output.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    input_diff = input_diff.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
    update_diff = update_diff.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))

    if verbose:
        print("dtype: {}".format(dtype))
        print("device_tag: {}".format(device_tag))
        print("input: {}\n{}\n".format(input.shape, input))
        print("output: {}\n{}\n".format(output.shape, output))

    def _make_diff_watcher(shape):
        def _watch_diff(diff: otp.Numpy):
            if shape == input_diff.shape:
                test_case.assertTrue(np.array_equal(diff, input_diff))
            elif shape == update_diff.shape:
                test_case.assertTrue(np.array_equal(diff, update_diff))

        return _watch_diff

    flow.clear_default_session()
    func_cfg = flow.FunctionConfig()
    func_cfg.default_data_type(dtype)
    func_cfg.default_placement_scope(flow.scope.placement(device_tag, "0:0"))
    slice_func = _make_slice_update_grad_func(
        slice_args, input.shape, update.shape, _make_diff_watcher, dtype, func_cfg
    )

    ret = slice_func(input, update)
    test_case.assertTrue(np.array_equal(ret, output))


@flow.unittest.skip_unless_1n1d()
class TestSliceV2(flow.unittest.TestCase):
    def test_slice_base(test_case):
        input = np.random.rand(10)
        slice_args = [[(1, 7, 2)]]
        outputs = [input[1:7:2]]

        arg_dict = collections.OrderedDict()
        arg_dict["dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]
        arg_dict["device_tag"] = ["cpu", "gpu"]
        # arg_dict["verbose"] = [True]
        for kwarg in test_util.GenArgDict(arg_dict):
            _test_slice(test_case, input, slice_args, outputs, **kwarg)

    def test_slice_into_two_parts(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [
            [(None, None, None), (0, 2, None), (None, None, None)],
            [(None, None, None), (2, None, None), (None, None, None)],
        ]
        outputs = [input[:, 0:2, :], input[:, 2:, :]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_at_first_dim(test_case):
        input = np.random.rand(4, 5, 4)
        slice_args = [[(2, None, None)]]
        outputs = [input[2:None, :, :]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_at_two_dims(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (0, 2, None), (2, None, None)]]
        outputs = [input[:, 0:2, 2:]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_with_collapse_dims(test_case):
        input = np.random.rand(2, 5, 4, 4, 3)
        slice_args = [
            [
                (None, None, None),
                (0, 2, None),
                (None, None, None),
                (None, None, None),
                (1, None, None),
            ]
        ]
        outputs = [input[:, 0:2, :, :, 1:]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_with_step_two(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (1, None, 2)]]
        outputs = [input[:, 1::2, :]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_at_two_dim_with_step_more_than_one(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (1, None, 3), (None, None, 2)]]
        outputs = [input[:, 1::3, ::2]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_with_neg_start(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (-4, None, None)]]
        outputs = [input[:, -4:, :]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_with_neg_stop(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (None, -2, None)]]
        outputs = [input[:, :-2, :]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_with_neg_step(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (None, None, -1)]]
        outputs = [input[:, ::-1, :]]
        _test_slice(test_case, input, slice_args, outputs)

    def test_slice_with_neg_step_two(test_case):
        input = np.random.rand(2, 5, 4)
        slice_args = [[(None, None, None), (-1, 1, -2)]]
        outputs = [input[:, -1:1:-2, :]]
        _test_slice(test_case, input, slice_args, outputs)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_slice_with_float16(test_case):
        input = np.random.rand(10).astype(np.float32)
        slice_args = [[(2, 7, None)]]
        outputs = [input[2:7]]

        flow.clear_default_session()
        flow.config.gpu_device_num(1)
        slice_func = _make_slice_with_fp16_func(slice_args, input.shape)
        of_outputs = slice_func(input)
        # print("outputs[0]:\n{}".format(outputs[0]))
        # print("of_outputs[0]:\n{}".format(of_outputs[0]))
        test_case.assertTrue(
            np.allclose(outputs[0], of_outputs[0], rtol=1e-03, atol=1e-04)
        )

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_base(test_case):
        input = np.random.rand(2, 4, 4)
        slice_args = [[(None, None, None), (1, None, None)]]
        outputs = [input[:, 1:, :]]

        arg_dict = collections.OrderedDict()
        arg_dict["dtype"] = [
            flow.uint8,
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float32,
            flow.float64,
        ]
        arg_dict["device_tag"] = ["cpu", "gpu"]
        for kwarg in test_util.GenArgDict(arg_dict):
            _test_slice_dynamic(
                test_case, input, slice_args, outputs, static_shape=(2, 5, 5), **kwarg
            )

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_at_two_dims(test_case):
        input = np.random.rand(2, 3, 2, 2)
        slice_args = [
            [(None, None, None), (2, None, None), (None, None, None), (1, None, None)]
        ]
        outputs = [input[:, 2:, :, 1:]]
        _test_slice_dynamic(
            test_case, input, slice_args, outputs, static_shape=(2, 5, 3, 3)
        )

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_at_first_dim_and_last_dim(test_case):
        input = np.random.rand(3, 6, 3, 3)
        slice_args = [
            [(1, None, None), (None, None, None), (None, None, None), (1, None, None)]
        ]
        outputs = [input[1:, :, :, 1:]]
        _test_slice_dynamic(
            test_case, input, slice_args, outputs, static_shape=(4, 5, 5, 3)
        )

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_neg_start(test_case):
        input = np.random.rand(2, 10)
        slice_args = [[(None, None, None), (-5, None, None)]]
        outputs = [input[:, -5:]]
        _test_slice_dynamic(test_case, input, slice_args, outputs, static_shape=(3, 7))

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_neg_step(test_case):
        input = np.random.rand(2, 10)
        slice_args = [[(None, None, None), (None, -5, -1)]]
        outputs = [input[:, :-5:-1]]
        _test_slice_dynamic(test_case, input, slice_args, outputs, static_shape=(3, 7))

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_anomaly(test_case):
        input = np.random.rand(4, 7)
        slice_args = [[(None, None, None), (2, None, None)]]
        outputs = [input[:, 2:]]
        _test_slice_dynamic(test_case, input, slice_args, outputs, static_shape=(5, 6))

    # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
    @unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
    def test_slice_dynamic_empty_blob(test_case):
        input = np.random.rand(5, 0, 5)
        slice_args = [[(None, None, None), (None, None, None), (2, 3, None)]]
        outputs = [input[:, :, 2:3]]
        _test_slice_dynamic(
            test_case, input, slice_args, outputs, static_shape=(8, 2, 10)
        )

    """This test case will raise fatal error, error infomation is like below:
    F0808 00:20:19.768465 23960 user_kernel.cpp:451] Check failed: shape_view.elem_cnt() <= static_shape.elem_cnt() (12 vs. 9)
    InferShape of OpKernel (op_type_name: slice, op_name: SliceDynamic_0) raise error,
    output arg's (name: y, index: 0) runtime shape (2,6) surpass the limit of static shape (3,3)
    *** Check failure stack trace: ***
    ...
    The reason is the dismatch between static slice (for memory) and dynamic slice (real slice)
    The result shape of slice [:, 3:-1] for static shape (3, 7) is (3, 3)
    which indicate that blob has prod(3, 3) memory limit,
    and the result shape of slice [:, 3:-1] for dynamic shape (2, 10) is (2, 6)
    which will cause blob to be out of memory limit.
    """
    # def test_slice_dynamic_dismatch(test_case):
    #     input = np.random.rand(2, 10)
    #     slice_args = [[(None, None, None), (3, -1, None)]]
    #     outputs = [input[:, 3:-1]]
    #     _test_slice_dynamic(test_case, input, slice_args, outputs, static_shape=(3, 7))

    """
    static shape after slice is (5, 4)
    dynamic shape after slice is (4, 5)
    static shape after slice is (5, 3)
    dynamic shape after slice is (4, 4)
    """
    # def test_slice_dynamic_anomaly_failed(test_case):
    #     input = np.random.rand(4, 7)
    #     slice_args = [[(None, None, None), (3, None, None)]]
    #     outputs = [input[:, 3:]]
    #     _test_slice_dynamic(test_case, input, slice_args, outputs, static_shape=(5, 6))

    def test_slice_with_grad(test_case):
        input = np.random.rand(2, 5, 4)
        slice_tup_list = [(None, None, None), (2, -2, None)]
        output = input[:, 2:-2, :]
        diff = np.zeros(input.shape, dtype=input.dtype)
        diff[:, 2:-2, :] = 1

        arg_dict = collections.OrderedDict()
        arg_dict["dtype"] = [flow.float32, flow.float64]
        arg_dict["device_tag"] = ["cpu", "gpu"]
        arg_dict["verbose"] = [False]
        for kwarg in test_util.GenArgDict(arg_dict):
            _test_slice_with_grad(
                test_case, input, slice_tup_list, output, diff, **kwarg
            )

    def test_slice_update(test_case):
        input = np.random.rand(10, 5, 4)
        update = input[5:, :-1, ::2]
        update = np.random.rand(*update.shape)
        output = np.copy(input)
        output[5:, :-1, ::2] = update
        slice_tup_list = [(5, None, None), (None, -1, None), (None, None, 2)]

        arg_dict = collections.OrderedDict()
        arg_dict["dtype"] = [flow.float32, flow.float64]
        arg_dict["device_tag"] = ["cpu", "gpu"]
        arg_dict["verbose"] = [False]
        for kwarg in test_util.GenArgDict(arg_dict):
            _test_slice_update(
                test_case, input, update, slice_tup_list, output, **kwarg
            )

    def test_slice_update_grad(test_case):
        input = np.random.rand(2, 7)
        update = input[:, 1:4]
        update = np.random.rand(*update.shape)
        update_diff = np.ones(update.shape)
        input_diff = np.ones(input.shape)
        input_diff[:, 1:4] = 0
        output = np.copy(input)
        output[:, 1:4] = update
        slice_tup_list = [(None, None, None), (1, 4, None)]

        arg_dict = collections.OrderedDict()
        arg_dict["dtype"] = [flow.float32, flow.float64]
        arg_dict["device_tag"] = ["cpu", "gpu"]
        arg_dict["verbose"] = [False]
        for kwarg in test_util.GenArgDict(arg_dict):
            _test_slice_update_grad(
                test_case,
                input,
                update,
                slice_tup_list,
                output,
                input_diff,
                update_diff,
                **kwarg
            )


if __name__ == "__main__":
    unittest.main()
