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
import oneflow as flow
import numpy as np
import os
import random

import oneflow.typing as oft


def _test_input_ndarray_contiguous(test_case, shape):
    assert len(shape) > 1
    more_than_one_dim_list = []
    for axis, dim in enumerate(shape[1:], 1):
        if dim > 1:
            more_than_one_dim_list.append((axis, dim))
    assert len(more_than_one_dim_list) > 0

    input = np.random.rand(*shape).astype(np.single)
    rand_axis = random.choice(more_than_one_dim_list)[0]
    rand_dim_slice_start = random.randrange(0, input.shape[rand_axis] - 1)
    rand_dim_slice_stop = random.randrange(
        rand_dim_slice_start + 1, input.shape[rand_axis]
    )
    slice_list = []
    for axis in range(input.ndim):
        if axis == rand_axis:
            slice_list.append(slice(rand_dim_slice_start, rand_dim_slice_stop))
        else:
            slice_list.append(slice(None))

    slice_input = input[tuple(slice_list)]
    assert (
        not slice_input.data.contiguous
    ), "shape: {}\nslice axis: {}, start~stop: {}~{}\nsliced shape: {}\nflags:\n{}".format(
        shape,
        rand_axis,
        rand_dim_slice_start,
        rand_dim_slice_stop,
        slice_input.shape,
        slice_input.flags,
    )

    flow.clear_default_session()

    @flow.global_function()
    def foo_job(
        x_def: oft.Numpy.Placeholder(shape=slice_input.shape, dtype=flow.float)
    ):
        y = x_def + flow.constant(1.0, shape=(1,), dtype=flow.float)
        return y

    ret = foo_job(slice_input).get()
    # print(ret.numpy().flags)
    test_case.assertTrue(ret.numpy().data.c_contiguous)
    test_case.assertTrue(np.array_equal(ret.numpy(), slice_input + 1.0))


# TODO: system op need manaully register blob_object in default_blob_register or bw_blob_register
# def test_eager_system_op(test_case):

#     flow.clear_default_session()
#     flow.enable_eager_execution()

#     input = np.random.rand(5, 4).astype(np.single)

#     @flow.global_function()
#     # def foo_job(x_def: oft.Numpy.Placeholder(shape=(5, 4), dtype=flow.float)):
#     def foo_job():
#         x = flow.constant(1, shape=(5, 4), dtype=flow.float)
#         y = flow.identity(x)
#         print([y.numpy(i) for i in range(y.numpy_size())])
#         # y = flow.identity(x_def)
#         # return y

#     foo_job()
#     # ret = foo_job(input).get()
#     # test_case.assertTrue(np.allclose(input, ret.numpy()))
@flow.unittest.skip_unless_1n1d()
class TestGlobalFunctionInputOutput(flow.unittest.TestCase):
    def test_lazy_input_output(test_case):
        flow.clear_default_session()
        flow.enable_eager_execution(False)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo_job(input_def: oft.Numpy.Placeholder(shape=(2, 5))):
            var = flow.get_variable(
                name="var",
                shape=(2, 5),
                dtype=flow.float,
                initializer=flow.ones_initializer(),
            )
            input_def = flow.cast_to_current_logical_view(input_def)
            var = flow.cast_to_current_logical_view(var)
            output = var + input_def
            return output

        input = np.arange(10).reshape(2, 5).astype(np.single)
        ret = foo_job(input).get()
        output = input + np.ones(shape=(2, 5), dtype=np.single)
        test_case.assertTrue(np.array_equal(output, ret.numpy()))

    def test_eager_output(test_case):

        flow.clear_default_session()
        flow.enable_eager_execution()

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo_job():
            x = flow.constant(1, shape=(2, 5), dtype=flow.float)
            return x

        ret = foo_job().get()
        test_case.assertTrue(
            np.array_equal(np.ones(shape=(2, 5), dtype=np.single), ret.numpy_list()[0])
        )

    def test_eager_multi_output(test_case):

        flow.clear_default_session()
        flow.enable_eager_execution()

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo_job():
            x = flow.constant(1, shape=(2, 5), dtype=flow.float)
            y = flow.get_variable(
                name="var",
                shape=(64, 4),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            return x, y

        x, y = foo_job().get()
        test_case.assertTrue(
            np.array_equal(np.ones(shape=(2, 5), dtype=np.single), x.numpy_list()[0])
        )
        test_case.assertTrue(
            np.array_equal(np.zeros(shape=(64, 4), dtype=np.single), y.numpy())
        )

    def test_eager_input(test_case):

        flow.clear_default_session()
        flow.enable_eager_execution()

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        input = np.random.rand(2, 5).astype(np.single)
        output = np.maximum(input, 0)

        @flow.global_function(function_config=func_config)
        def foo_job(x_def: oft.ListNumpy.Placeholder(shape=(2, 5), dtype=flow.float)):
            y = flow.math.relu(x_def)
            test_case.assertTrue(np.allclose(y.numpy(0), output))

        foo_job([input])

    def test_eager_input_fixed(test_case):

        flow.clear_default_session()
        flow.enable_eager_execution()

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        input = np.arange(10).astype(np.single)
        output = input + 1.0

        @flow.global_function(function_config=func_config)
        def foo_job(x_def: oft.Numpy.Placeholder(shape=(10,), dtype=flow.float)):
            y = x_def + flow.constant(1.0, shape=(1,), dtype=flow.float)
            test_case.assertTrue(np.allclose(y.numpy(0), output))

        foo_job(input)

    def test_eager_multi_input(test_case):

        flow.clear_default_session()
        flow.enable_eager_execution()

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        input_1 = np.random.rand(3, 4).astype(np.single)
        input_2 = np.array([2]).astype(np.single)
        output = input_1 * input_2

        @flow.global_function(function_config=func_config)
        def foo_job(
            x_def: oft.ListNumpy.Placeholder(shape=(3, 4), dtype=flow.float),
            y_def: oft.ListNumpy.Placeholder(shape=(1,), dtype=flow.float),
        ):
            y = x_def * y_def
            test_case.assertTrue(np.allclose(y.numpy(0), output))

        foo_job([input_1], [input_2])

    def test_eager_input_output(test_case):

        flow.clear_default_session()
        flow.enable_eager_execution()

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        input = np.random.rand(5, 4).astype(np.single)
        output = input * 2.0

        @flow.global_function(function_config=func_config)
        def foo_job(x_def: oft.ListNumpy.Placeholder(shape=(5, 4), dtype=flow.float)):
            y = x_def * flow.constant(2.0, shape=(1,), dtype=flow.float)
            return y

        ret = foo_job([input]).get()
        test_case.assertTrue(np.allclose(output, ret.numpy_list()[0]))

    def test_input_ndarray_contiguous(test_case):
        _test_input_ndarray_contiguous(test_case, (10, 20, 30))


if __name__ == "__main__":
    unittest.main()
