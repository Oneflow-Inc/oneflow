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
import oneflow.typing as oft
import tensorflow as tf
import test_global_storage
import random
import math
import unittest
import os

from test_util import GenArgList, type_name_to_flow_type
from collections import OrderedDict

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, x_shape, y_shape, dtype, axis):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def ConcatJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            y = flow.get_variable(
                "y",
                shape=y_shape,
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            x = flow.cast_to_current_logical_view(x)
            y = flow.cast_to_current_logical_view(y)
            loss = flow.concat([x, y], axis)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(y, test_global_storage.Setter("y"))
            flow.watch_diff(y, test_global_storage.Setter("y_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    of_out = ConcatJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        y = tf.Variable(test_global_storage.Get("y"))
        tf_out = tf.concat([x, y], axis)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    tf_y_diff = tape.gradient(tf_out, y, loss_diff)

    assert np.array_equal(of_out.numpy(), tf_out.numpy())
    assert np.array_equal(test_global_storage.Get("x_diff"), tf_x_diff.numpy())
    assert np.array_equal(test_global_storage.Get("y_diff"), tf_y_diff.numpy())


def _of_dynamic_concat(
    inputs,
    input_static_shape,
    axis,
    device_type,
    watch_cb=None,
    make_watch_diff_cb=None,
):
    assert isinstance(inputs, (list, tuple))
    assert len(inputs) >= 2
    assert callable(make_watch_diff_cb)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))

    @flow.global_function(type="train", function_config=func_config)
    def dynamic_concat_job(
        input_0_def: oft.ListNumpy.Placeholder(
            shape=input_static_shape, dtype=flow.float
        ),
        input_1_def: oft.ListNumpy.Placeholder(
            shape=input_static_shape, dtype=flow.float
        ),
    ):
        var_0 = flow.get_variable(
            "Var0",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.constant_initializer(value=1, dtype=flow.float),
            trainable=True,
        )
        var_1 = flow.get_variable(
            "Var1",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.constant_initializer(value=1, dtype=flow.float),
            trainable=True,
        )
        var_0 = flow.cast_to_current_logical_view(var_0)
        var_1 = flow.cast_to_current_logical_view(var_1)
        input_0_def = flow.cast_to_current_logical_view(input_0_def)
        input_1_def = flow.cast_to_current_logical_view(input_1_def)
        if callable(watch_cb):
            flow.watch(var_0, watch_cb)
            flow.watch(var_1, watch_cb)
            flow.watch(flow.identity(input_0_def), watch_cb)
            flow.watch(flow.identity(input_1_def), watch_cb)

        var_0 = var_0 * input_0_def
        var_1 = var_1 * input_1_def
        if callable(watch_cb):
            flow.watch(var_0, watch_cb)
            flow.watch(var_1, watch_cb)

        result = flow.concat(
            [var_0, var_1], axis=axis, max_dim_size=input_static_shape[axis]
        )
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
        ).minimize(result)
        flow.watch_diff(var_0, make_watch_diff_cb(0))
        flow.watch_diff(var_1, make_watch_diff_cb(1))
        return result

    ret = dynamic_concat_job([inputs[0]], [inputs[1]]).get()
    return ret.numpy(0)


def _rand_part_range(start, stop, part_num):
    part_size = math.ceil((stop - start) / part_num)
    begin = start
    for i in range(part_num):
        end = part_size * (i + 2)
        end = random.randrange(begin + 1, min(end, stop - (part_num - i - 1)))
        yield (begin, end)
        begin = end


def _slice(input, axis, start, stop):
    slice_list = []
    for i in range(input.ndim):
        if i == axis:
            slice_list.append(slice(start, stop))
        else:
            slice_list.append(slice(None))

    return input[tuple(slice_list)]


def _rand_inputs(shape, split_axis, part_num):
    entire_input = np.random.rand(*shape).astype(np.single)
    inputs = []
    last_stop = 0
    for start, stop in _rand_part_range(0, shape[split_axis], part_num):
        last_stop = stop
        input_slice = _slice(entire_input, split_axis, start, stop)
        inputs.append(input_slice)

    return _slice(entire_input, split_axis, 0, last_stop), inputs


def _test_dynamic_concat(test_case, shape, axis, device_type, verbose=False):
    assert axis >= 0 and axis < len(shape)

    output, inputs = _rand_inputs(shape, axis, 2)

    def print_blob(blob):
        print(blob.numpy(0), blob.numpy(0).shape)

    def make_watch_diff_cb(input_idx):
        def watch_diff_cb(blob):
            test_case.assertTrue(
                np.array_equal(
                    blob.numpy(0),
                    np.ones(shape=inputs[input_idx].shape, dtype=np.single),
                )
            )

        return watch_diff_cb

    of_output = _of_dynamic_concat(
        inputs,
        tuple(shape),
        axis,
        device_type,
        print_blob if verbose else None,
        make_watch_diff_cb,
    )

    if verbose:
        print("inputs shapes:", [input.shape for input in inputs])
        print("output shape:", output.shape)
        print("of_output shape:", of_output.shape)
        print("output:\n", output)
        print("of_output:\n", of_output)

    test_case.assertTrue(np.array_equal(of_output, output))


def _test_static_concat(test_case, shape, axis):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    def compare_var_diff(var_blob):
        test_case.assertTrue(
            np.array_equal(var_blob.numpy(), np.ones(shape=shape, dtype=np.single))
        )

    @flow.global_function(type="train", function_config=func_config)
    def static_concat_job(
        input_0_def: oft.Numpy.Placeholder(shape=shape, dtype=flow.float),
        input_1_def: oft.Numpy.Placeholder(shape=shape, dtype=flow.float),
    ):
        var = flow.get_variable(
            "var",
            shape=shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
            trainable=True,
        )
        concated = flow.concat([input_0_def, input_1_def, var], axis=axis)
        test_case.assertTrue(not concated.is_dynamic)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(concated)
        flow.watch_diff(var, compare_var_diff)
        return var, concated

    inputs = []
    for i in range(2):
        inputs.append(np.random.rand(*shape).astype(np.single))
    var, concated = static_concat_job(inputs[0], inputs[1]).get()
    test_case.assertTrue(
        np.array_equal(
            np.concatenate([inputs[0], inputs[1], var.numpy()], axis=axis,),
            concated.numpy(),
        )
    )


def _test_hybrid_concat(
    test_case, static_shape, axis, max_dim_size=None, verbose=False
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())

    def compare_var_diff(var_blob):
        test_case.assertTrue(
            np.array_equal(
                var_blob.numpy(), np.ones(shape=static_shape, dtype=np.single)
            )
        )

    rand_sub_shape = list(static_shape).copy()
    rand_sub_shape[axis] = random.randrange(1, static_shape[axis])
    rand_sub_shape = tuple(rand_sub_shape)

    @flow.global_function(type="train", function_config=func_config)
    def hybrid_concat_job(
        input_0_def: oft.ListNumpy.Placeholder(shape=static_shape, dtype=flow.float),
        input_1_def: oft.ListNumpy.Placeholder(shape=static_shape, dtype=flow.float),
    ):
        var = flow.get_variable(
            "var",
            shape=static_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
            trainable=True,
        )
        constant = flow.constant(1.0, dtype=flow.float, shape=rand_sub_shape)
        inputs = [
            flow.cast_to_current_logical_view(input)
            for input in [var, input_0_def, input_1_def, constant]
        ]
        concated = flow.concat(inputs, axis=axis, max_dim_size=max_dim_size,)
        if verbose:
            print("concated static shape:", concated.shape)

        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(concated)
        flow.watch_diff(var, compare_var_diff)

        if max_dim_size is None:
            test_case.assertTrue(
                concated.shape[axis] == (static_shape[axis] * 3 + rand_sub_shape[axis])
            )
        else:
            test_case.assertTrue(concated.shape[axis] == max_dim_size)

        return var, concated

    output, inputs = _rand_inputs(static_shape, axis, 2)
    if verbose:
        print("static_shape:", static_shape)
        print("input_0 shape:", inputs[0].shape)
        print("input_1 shape:", inputs[1].shape)
        print("output shape:", output.shape)
        print("rand_sub_shape:", rand_sub_shape)

    var, concated = hybrid_concat_job([inputs[0]], [inputs[1]]).get()
    if verbose:
        print("var shape:", var.numpy().shape)
        print("concated shape:", concated.numpy(0).shape)

    test_case.assertTrue(
        np.array_equal(
            np.concatenate(
                [var.numpy(), output, np.ones(shape=rand_sub_shape, dtype=np.single)],
                axis=axis,
            ),
            concated.numpy(0),
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestConcat(flow.unittest.TestCase):
    def test_concat(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [(10, 20, 30)]
        arg_dict["y_shape"] = [(10, 20, 30)]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["axis"] = [0, 1, 2]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dynamic_concat_case_0(test_case):
        _test_dynamic_concat(test_case, (64, 4), 0, "gpu")

    def test_dynamic_concat_case_1(test_case):
        _test_dynamic_concat(test_case, (2, 10), 1, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dynamic_concat_case_2(test_case):
        _test_dynamic_concat(test_case, (4, 7, 128), 2, "gpu")

    def test_dynamic_concat_case_3(test_case):
        _test_dynamic_concat(test_case, (16,), 0, "cpu")

    def test_static_concat_case_0(test_case):
        _test_static_concat(test_case, (10, 7), 0)

    def test_static_concat_case_1(test_case):
        _test_static_concat(test_case, (3, 8, 4), 1)

    def test_hybrid_concat_case_0(test_case):
        _test_hybrid_concat(test_case, (64, 4), 0)

    def test_hybrid_concat_case_1(test_case):
        _test_hybrid_concat(test_case, (10,), 0, 30)

    def test_hybrid_concat_case_2(test_case):
        _test_hybrid_concat(test_case, (10, 7, 5), 1, 21)


if __name__ == "__main__":
    unittest.main()
