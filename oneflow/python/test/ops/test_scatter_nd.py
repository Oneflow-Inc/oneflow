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
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList
import oneflow.typing as oft
import unittest
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _random_inputs(
    params_shape, indices_shape, updates_shape, allow_duplicate_index=True
):
    params = np.random.rand(*params_shape).astype(np.float32)
    updates = np.random.rand(*updates_shape).astype(np.float32)
    indices = []
    indices_rows = np.prod(indices_shape[:-1])
    indices_cols = indices_shape[-1]
    for col in range(indices_cols):
        if allow_duplicate_index is False and indices_rows <= params_shape[col]:
            rand_indices = np.arange(params_shape[col], dtype=np.int32)
            np.random.shuffle(rand_indices)
            indices_col = rand_indices[:indices_rows].reshape(indices_shape[:-1])
        else:
            indices_col = np.random.randint(
                low=0, high=params_shape[col], size=(indices_rows,), dtype=np.int32
            ).reshape(indices_shape[:-1])
        indices.append(indices_col)
    indices = np.stack(indices, axis=len(indices_shape) - 1)

    if allow_duplicate_index is False:
        existing_nd_index_set = set()
        for nd_index in indices.reshape(-1, indices.shape[-1]):
            nd_index_str = "(" + ",".join(map(str, nd_index)) + ")"
            assert (
                nd_index_str not in existing_nd_index_set
            ), "random generated duplicate nd index {}".format(nd_index_str)
            existing_nd_index_set.add(nd_index_str)

    return params, updates, indices


def _make_scatter_nd_fn(indices, updates, shape, device_type, mirrored, compare_fn):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    if mirrored:
        func_config.default_logical_view(flow.scope.mirrored_view())
    else:
        func_config.default_logical_view(flow.scope.consistent_view())

    def do_scatter_nd(indices_blob, updates_blob):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "updates",
                shape=updates.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            x = flow.cast_to_current_logical_view(x)
            x = x + updates_blob
            y = flow.scatter_nd(indices_blob, x, shape)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(y)
        flow.watch_diff(x, compare_fn)
        return y

    if mirrored:

        @flow.global_function(type="train", function_config=func_config)
        def scatter_nd_fn(
            indices_def: oft.ListNumpy.Placeholder(indices.shape, dtype=flow.int32),
            updates_def: oft.ListNumpy.Placeholder(updates.shape, dtype=flow.float),
        ):
            return do_scatter_nd(indices_def, updates_def)

    else:

        @flow.global_function(type="train", function_config=func_config)
        def scatter_nd_fn(
            indices_def: oft.Numpy.Placeholder(indices.shape, dtype=flow.int32),
            updates_def: oft.Numpy.Placeholder(updates.shape, dtype=flow.float),
        ):
            return do_scatter_nd(indices_def, updates_def)

    return scatter_nd_fn


def _compare_scatter_nd_with_tf(
    test_case,
    device_type,
    params_shape,
    indices_shape,
    updates_shape,
    mirrored=False,
    verbose=False,
):
    _, updates, indices = _random_inputs(params_shape, indices_shape, updates_shape)

    indices_const = tf.constant(indices)
    with tf.GradientTape() as t:
        x = tf.Variable(updates)
        y = tf.scatter_nd(indices_const, x, params_shape)

    dy_dx = t.gradient(y, x)

    if mirrored:

        def compare_dy(params_grad):
            test_case.assertTrue(
                np.array_equal(dy_dx.numpy(), params_grad.numpy_list()[0])
            )

    else:

        def compare_dy(params_grad):
            test_case.assertTrue(np.array_equal(dy_dx.numpy(), params_grad.numpy()))

    scatter_nd_fn = _make_scatter_nd_fn(
        indices, updates, params_shape, device_type, mirrored, compare_dy
    )

    if mirrored:
        of_y = scatter_nd_fn([indices], [updates]).get().numpy_list()[0]
    else:
        of_y = scatter_nd_fn(indices, updates).get().numpy()

    if verbose is True:
        print("device_type:", device_type)
        print("indices:", indices)
        print("updates:", updates)
        print("tf_params:", y.numpy())
        print("of_params:", of_y)

    test_case.assertTrue(np.allclose(y.numpy(), of_y))


def _compare_scatter_nd_update_with_tf(
    test_case,
    device_type,
    params_shape,
    indices_shape,
    updates_shape,
    allow_duplicate_index=False,
    verbose=False,
):
    params, updates, indices = _random_inputs(
        params_shape, indices_shape, updates_shape, allow_duplicate_index
    )

    x_const = tf.constant(params)
    y_const = tf.constant(updates)
    i_const = tf.constant(indices)
    with tf.GradientTape() as t1:
        x = tf.Variable(params)
        z1 = tf.tensor_scatter_nd_update(x, i_const, y_const)
    dz_dx = t1.gradient(z1, x)

    with tf.GradientTape() as t2:
        y = tf.Variable(updates)
        z2 = tf.tensor_scatter_nd_update(x_const, i_const, y)
    dz_dy = t2.gradient(z2, y)

    test_case.assertTrue(np.allclose(z1.numpy(), z2.numpy()))

    def compare_dz_dx(params_grad):
        test_case.assertTrue(np.allclose(dz_dx.numpy(), params_grad.numpy()))

    def compare_dz_dy(updates_grad):
        test_case.assertTrue(np.allclose(dz_dy.numpy(), updates_grad.numpy()))

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(type="train", function_config=func_config)
    def scatter_nd_update_grad_fn(
        x_def: oft.Numpy.Placeholder(params.shape, dtype=flow.float),
        indices_def: oft.Numpy.Placeholder(indices.shape, dtype=flow.int32),
        y_def: oft.Numpy.Placeholder(updates.shape, dtype=flow.float),
    ):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "params",
                shape=params.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            y = flow.get_variable(
                "updates",
                shape=updates.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            x = x + x_def
            y = y + y_def
            z = flow.tensor_scatter_nd_update(x, indices_def, y)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(z)

        flow.watch_diff(x, compare_dz_dx)
        flow.watch_diff(y, compare_dz_dy)
        return z

    of_z = scatter_nd_update_grad_fn(params, indices, updates).get()

    if verbose is True:
        print("device_type:", device_type)
        print("x:", params)
        print("y:", updates)
        print("indices:", indices)
        print("tf_z:", z1.numpy())
        print("of_z:", of_z.numpy())

    test_case.assertTrue(np.allclose(z1.numpy(), of_z.numpy()))


def _of_tensor_scatter_nd_add(
    params,
    indices,
    updates,
    device_type,
    mirrored,
    params_grad_watcher,
    updates_grad_watcher,
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def do_tensor_scatter_nd_add(params_blob, indices_blob, updates_blob):
        with flow.scope.placement(device_type, "0:0"):
            params_var = flow.get_variable(
                "params",
                shape=params_blob.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            updates_var = flow.get_variable(
                "updates",
                shape=updates_blob.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            params_var = flow.cast_to_current_logical_view(params_var)
            params_blob = flow.cast_to_current_logical_view(params_blob)
            updates_blob = flow.cast_to_current_logical_view(updates_blob)
            updates_var = flow.cast_to_current_logical_view(updates_var)
            params_var = params_var + params_blob
            updates_var = updates_var + updates_blob
            out = flow.tensor_scatter_nd_add(params_var, indices_blob, updates_var)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(out)

        flow.watch_diff(params_var, params_grad_watcher)
        flow.watch_diff(updates_var, updates_grad_watcher)
        return out

    if mirrored:
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(type="train", function_config=func_config)
        def tensor_scatter_nd_add_fn(
            params_def: oft.ListNumpy.Placeholder(params.shape, dtype=flow.float),
            indices_def: oft.ListNumpy.Placeholder(indices.shape, dtype=flow.int32),
            updates_def: oft.ListNumpy.Placeholder(updates.shape, dtype=flow.float),
        ):
            return do_tensor_scatter_nd_add(params_def, indices_def, updates_def)

        return (
            tensor_scatter_nd_add_fn([params], [indices], [updates])
            .get()
            .numpy_list()[0]
        )

    else:
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(type="train", function_config=func_config)
        def tensor_scatter_nd_add_fn(
            params_def: oft.Numpy.Placeholder(params.shape, dtype=flow.float),
            indices_def: oft.Numpy.Placeholder(indices.shape, dtype=flow.int32),
            updates_def: oft.Numpy.Placeholder(updates.shape, dtype=flow.float),
        ):
            return do_tensor_scatter_nd_add(params_def, indices_def, updates_def)

        return tensor_scatter_nd_add_fn(params, indices, updates).get().numpy()


def _compare_tensor_scatter_nd_add_with_tf(
    test_case, params_shape, indices_shape, updates_shape, device_type, mirrored
):
    params, updates, indices = _random_inputs(
        params_shape, indices_shape, updates_shape, True
    )

    params_const = tf.constant(params)
    indices_const = tf.constant(indices)
    updates_const = tf.constant(updates)
    with tf.GradientTape() as t1:
        params_var = tf.Variable(params)
        tf_out1 = tf.tensor_scatter_nd_add(params_var, indices_const, updates_const)
    tf_params_grad = t1.gradient(tf_out1, params_var)

    with tf.GradientTape() as t2:
        updates_var = tf.Variable(updates)
        tf_out2 = tf.tensor_scatter_nd_add(params_const, indices_const, updates_var)
    tf_updates_grad = t2.gradient(tf_out2, updates_var)

    test_case.assertTrue(np.allclose(tf_out1.numpy(), tf_out2.numpy()))

    def compare_params_grad(of_params_grad):
        tf_params_grad_np = tf_params_grad.numpy()
        of_params_grad_np = (
            of_params_grad.numpy_list()[0] if mirrored else of_params_grad.numpy()
        )
        test_case.assertTrue(np.allclose(tf_params_grad_np, of_params_grad_np))

    def compare_updates_grad(of_updates_grad):
        tf_updates_grad_np = tf_updates_grad.numpy()
        of_updates_grad_np = (
            of_updates_grad.numpy_list()[0] if mirrored else of_updates_grad.numpy()
        )
        test_case.assertTrue(np.allclose(tf_updates_grad_np, of_updates_grad_np))

    of_out = _of_tensor_scatter_nd_add(
        params,
        indices,
        updates,
        device_type,
        mirrored,
        compare_params_grad,
        compare_updates_grad,
    )
    test_case.assertTrue(np.allclose(tf_out1.numpy(), of_out))


def _of_scatter_nd_dynamic_indices(
    indices, updates, indices_static_shape, updates_static_shape, params_shape
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def scatter_nd_fn(
        indices_def: oft.ListNumpy.Placeholder(indices_static_shape, dtype=flow.int32),
        updates_def: oft.ListNumpy.Placeholder(updates_static_shape, dtype=flow.float),
    ):
        with flow.scope.placement("gpu", "0:0"):
            return flow.scatter_nd(indices_def, updates_def, params_shape)

    return scatter_nd_fn([indices], [updates]).get().numpy_list()[0]


def _compare_scatter_nd_dynamic_indices_with_tf(
    test_case,
    indices_shape,
    updates_shape,
    indices_static_shape,
    updates_static_shape,
    params_shape,
):
    _, updates, indices = _random_inputs(params_shape, indices_shape, updates_shape)

    indices_const = tf.constant(indices)
    x = tf.Variable(updates)
    y = tf.scatter_nd(indices_const, x, params_shape)

    of_y = _of_scatter_nd_dynamic_indices(
        indices, updates, indices_static_shape, updates_static_shape, params_shape
    )
    test_case.assertTrue(np.allclose(y.numpy(), of_y))


def _of_tensor_scatter_nd_update_dynamic_indices(
    params, indices, updates, indices_static_shape, updates_static_shape
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def tensor_scatter_nd_update_fn(
        params_def: oft.ListNumpy.Placeholder(params.shape, dtype=flow.float),
        indices_def: oft.ListNumpy.Placeholder(indices_static_shape, dtype=flow.int32),
        updates_def: oft.ListNumpy.Placeholder(updates_static_shape, dtype=flow.float),
    ):
        with flow.scope.placement("gpu", "0:0"):
            return flow.tensor_scatter_nd_update(params_def, indices_def, updates_def)

    return (
        tensor_scatter_nd_update_fn([params], [indices], [updates])
        .get()
        .numpy_list()[0]
    )


def _compare_tensor_scatter_nd_update_dynamic_indices_with_tf(
    test_case,
    params_shape,
    indices_shape,
    updates_shape,
    indices_static_shape,
    updates_static_shape,
):
    params, updates, indices = _random_inputs(
        params_shape, indices_shape, updates_shape, False
    )

    i = tf.constant(indices)
    x = tf.Variable(params)
    y = tf.Variable(updates)
    z = tf.tensor_scatter_nd_update(x, i, y)

    of_z = _of_tensor_scatter_nd_update_dynamic_indices(
        params, indices, updates, indices_static_shape, updates_static_shape
    )
    test_case.assertTrue(np.allclose(z.numpy(), of_z))


def _of_tensor_scatter_nd_add_dynamic_indices(
    params, indices, updates, indices_static_shape, updates_static_shape
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def tensor_scatter_nd_add_fn(
        params_def: oft.ListNumpy.Placeholder(params.shape, dtype=flow.float),
        indices_def: oft.ListNumpy.Placeholder(indices_static_shape, dtype=flow.int32),
        updates_def: oft.ListNumpy.Placeholder(updates_static_shape, dtype=flow.float),
    ):
        with flow.scope.placement("gpu", "0:0"):
            return flow.tensor_scatter_nd_add(params_def, indices_def, updates_def)

    return (
        tensor_scatter_nd_add_fn([params], [indices], [updates]).get().numpy_list()[0]
    )


def _compare_tensor_scatter_nd_add_dynamic_indices_with_tf(
    test_case,
    params_shape,
    indices_shape,
    updates_shape,
    indices_static_shape,
    updates_static_shape,
):
    params, updates, indices = _random_inputs(
        params_shape, indices_shape, updates_shape
    )

    i = tf.constant(indices)
    x = tf.Variable(params)
    y = tf.Variable(updates)
    z = tf.tensor_scatter_nd_add(x, i, y)

    of_z = _of_tensor_scatter_nd_add_dynamic_indices(
        params, indices, updates, indices_static_shape, updates_static_shape
    )
    test_case.assertTrue(np.allclose(z.numpy(), of_z))


@flow.unittest.skip_unless_1n1d()
class TestScatterNd(flow.unittest.TestCase):
    def test_scatter_nd(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["params_shape"] = [(10,)]
        arg_dict["indices_shape"] = [(5, 1)]
        arg_dict["updates_shape"] = [(5,)]
        arg_dict["mirrored"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_with_tf(test_case, *arg)

    def test_scatter_nd_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(128,)]
        arg_dict["indices_shape"] = [(100, 1)]
        arg_dict["updates_shape"] = [(100,)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_with_tf(test_case, *arg)

    def test_scatter_nd_case_2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(32, 16, 4)]
        arg_dict["indices_shape"] = [(50, 2)]
        arg_dict["updates_shape"] = [(50, 4)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_with_tf(test_case, *arg)

    def test_scatter_nd_case_3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(24, 25, 32, 10, 12)]
        arg_dict["indices_shape"] = [(3, 4, 2)]
        arg_dict["updates_shape"] = [(3, 4, 32, 10, 12)]
        arg_dict["mirrored"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_with_tf(test_case, *arg)

    def test_scatter_nd_case_4(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(8,)]
        arg_dict["indices_shape"] = [(12, 1)]
        arg_dict["updates_shape"] = [(12,)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_with_tf(test_case, *arg)

    def test_scatter_nd_update(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["params_shape"] = [(10,)]
        arg_dict["indices_shape"] = [(5, 1)]
        arg_dict["updates_shape"] = [(5,)]
        arg_dict["allow_duplicate_index"] = [False]
        # arg_dict["verbose"] = [True]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_update_with_tf(test_case, *arg)

    def test_scatter_nd_update_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(256, 64)]
        arg_dict["indices_shape"] = [(128, 2)]
        arg_dict["updates_shape"] = [(128,)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_update_with_tf(test_case, *arg)

    def test_scatter_nd_update_case_2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(20, 10, 11, 3, 5)]
        arg_dict["indices_shape"] = [(2, 4, 3)]
        arg_dict["updates_shape"] = [(2, 4, 3, 5)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_update_with_tf(test_case, *arg)

    def test_scatter_nd_update_case_3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["params_shape"] = [(256, 4)]
        arg_dict["indices_shape"] = [(10, 25, 1)]
        arg_dict["updates_shape"] = [(10, 25, 4)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_update_with_tf(test_case, *arg)

    def test_tensor_scatter_nd_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(12,)]
        arg_dict["indices_shape"] = [(7, 1)]
        arg_dict["updates_shape"] = [(7,)]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["mirrored"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_add_with_tf(test_case, *arg)

    def test_tensor_scatter_nd_add_case1(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(38, 66, 9)]
        arg_dict["indices_shape"] = [(17, 2)]
        arg_dict["updates_shape"] = [(17, 9)]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["mirrored"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_add_with_tf(test_case, *arg)

    def test_tensor_scatter_nd_add_case2(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(2, 7, 19, 41, 33)]
        arg_dict["indices_shape"] = [(20, 9, 3)]
        arg_dict["updates_shape"] = [(20, 9, 41, 33)]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["mirrored"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_add_with_tf(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_scatter_nd_dynamic_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["indices_shape"] = [(12, 10, 2)]
        arg_dict["updates_shape"] = [(12, 10, 41, 33)]
        arg_dict["indices_static_shape"] = [(15, 10, 2)]
        arg_dict["updates_static_shape"] = [(15, 10, 41, 33)]
        arg_dict["params_shape"] = [(64, 22, 41, 33)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_dynamic_indices_with_tf(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_scatter_nd_empty_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["indices_shape"] = [(0, 1)]
        arg_dict["updates_shape"] = [(0, 14)]
        arg_dict["indices_static_shape"] = [(8, 1)]
        arg_dict["updates_static_shape"] = [(8, 14)]
        arg_dict["params_shape"] = [(10, 14)]
        for arg in GenArgList(arg_dict):
            _compare_scatter_nd_dynamic_indices_with_tf(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_scatter_nd_update_dynamic_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(32, 33, 4, 5)]
        arg_dict["indices_shape"] = [(12, 2)]
        arg_dict["updates_shape"] = [(12, 4, 5)]
        arg_dict["indices_static_shape"] = [(14, 2)]
        arg_dict["updates_static_shape"] = [(14, 4, 5)]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_update_dynamic_indices_with_tf(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_scatter_nd_update_empty_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(37, 14)]
        arg_dict["indices_shape"] = [(7, 0, 1)]
        arg_dict["updates_shape"] = [(7, 0, 14)]
        arg_dict["indices_static_shape"] = [(7, 5, 1)]
        arg_dict["updates_static_shape"] = [(7, 5, 14)]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_update_dynamic_indices_with_tf(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_scatter_nd_add_dynamic_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(2, 9, 7, 5, 4)]
        arg_dict["indices_shape"] = [(12, 5, 3)]
        arg_dict["updates_shape"] = [(12, 5, 5, 4)]
        arg_dict["indices_static_shape"] = [(15, 6, 3)]
        arg_dict["updates_static_shape"] = [(15, 6, 5, 4)]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_add_dynamic_indices_with_tf(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_scatter_nd_add_empty_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["params_shape"] = [(24, 30, 14)]
        arg_dict["indices_shape"] = [(0, 2)]
        arg_dict["updates_shape"] = [(0, 14)]
        arg_dict["indices_static_shape"] = [(11, 2)]
        arg_dict["updates_static_shape"] = [(11, 14)]
        for arg in GenArgList(arg_dict):
            _compare_tensor_scatter_nd_add_dynamic_indices_with_tf(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
