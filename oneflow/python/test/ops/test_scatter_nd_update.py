import oneflow as flow
import numpy as np
import tensorflow as tf
import os
from collections import OrderedDict
from test_util import GenArgList

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.enable_eager_execution()


def _random_inputs(params_shape, indices_shape, updates_shape, allow_duplicate_index=False):
    params = np.random.rand(*params_shape).astype(np.float32)
    updates = np.random.rand(*updates_shape).astype(np.float32)
    indices = []
    indices_rows = np.prod(indices_shape[:-1])
    indices_cols = indices_shape[-1]
    for col in range(indices_cols):
        # If values in params is to be updated more than once,
        # because there are duplicate entries in indices,
        # the order at which the updates happen for each value is undefined.
        if allow_duplicate_index:
            indices_col = np.random.randint(
                low=0, high=params_shape[col], size=(indices_rows,), dtype=np.int32
            ).reshape(indices_shape[:-1])
        else:
            assert indices_rows <= params_shape[col], "col=={},{} vs {}".format(
                col, indices_rows, params_shape[col]
            )
            rand_indices = np.arange(params_shape[col], dtype=np.int32)
            np.random.shuffle(rand_indices)
            indices_col = rand_indices[:indices_rows].reshape(indices_shape[:-1])
        indices.append(indices_col)
    indices = np.stack(indices, axis=len(indices_shape) - 1)
    return params, updates, indices


def _compare_scatter_nd_update_with_tf(
    test_case, device_type, params_shape, indices_shape, updates_shape
):
    params, updates, indices = _random_inputs(params_shape, indices_shape, updates_shape)

    x = tf.Variable(params)
    y = tf.Variable(updates)
    i = tf.Variable(indices)
    const_x = tf.constant(params)
    const_y = tf.constant(updates)

    with tf.GradientTape() as t1:
        z1 = tf.tensor_scatter_nd_update(x, i, const_y)
        dz_dx = t1.gradient(z1, x)

    with tf.GradientTape() as t2:
        # Cannot use tf.compat.v1.scatter_nd_update because it's ref input
        # and don't work correctly with gradient and it cannot calculate
        # updates grad
        z2 = tf.tensor_scatter_nd_update(const_x, i, y)
        dz_dy = t2.gradient(z2, y)

    test_case.assertTrue(np.allclose(z1.numpy(), z2.numpy()))

    def compare_dz_dx(params_grad):
        test_case.assertTrue(np.allclose(dz_dx.numpy(), params_grad.ndarray()))

    def compare_dz_dy(updates_grad):
        test_case.assertTrue(np.allclose(dz_dy.numpy(), updates_grad.ndarray()))

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.train.primary_lr(1e-3)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def scatter_nd_update_grad_fn(
        x_def=flow.FixedTensorDef(params.shape, dtype=flow.float),
        indices_def=flow.FixedTensorDef(indices.shape, dtype=flow.int32),
        y_def=flow.FixedTensorDef(updates.shape, dtype=flow.float),
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "params",
                shape=params.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                # distribute=flow.distribute.split(axis=0),
            )
            # x = x.with_split_distribute(axis=0)
            y = flow.get_variable(
                "updates",
                shape=updates.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                # distribute=flow.distribute.broadcast(),
            )
            x = x + x_def
            y = y + y_def
            z = flow.scatter_nd_update(x, indices_def, y)
            flow.losses.add_loss(z)

        flow.watch_diff(x, compare_dz_dx)
        flow.watch_diff(y, compare_dz_dy)
        return z

    check_point = flow.train.CheckPoint()
    check_point.init()
    of_z = scatter_nd_update_grad_fn(params, indices, updates).get()
    # print("device_type:", device_type)
    # print("x:", params)
    # print("y:", updates)
    # print("indices:", indices)
    # print("tf_z:", z1.numpy())
    # print("of_z:", of_z.ndarray())
    test_case.assertTrue(np.allclose(z1.numpy(), of_z.ndarray()))


def _compare_scatter_nd_update_mirrored_with_tf(test_case, params, indices, updates):
    tf_out = tf.tensor_scatter_nd_update(
        tf.Variable(params), tf.Variable(indices), tf.Variable(updates)
    ).numpy()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def scatter_nd_update_fn(
        input_def=flow.MirroredTensorDef(params.shape, dtype=flow.float),
        indices_def=flow.MirroredTensorDef(indices.shape, dtype=flow.int32),
        updates_def=flow.MirroredTensorDef(updates.shape, dtype=flow.float),
    ):
        with flow.device_prior_placement("gpu", "0:0"):
            return flow.scatter_nd_update(input_def, indices_def, updates_def)

    of_out = scatter_nd_update_fn([params], [indices], [updates]).get().ndarray_list()[0]
    # print("tf_out", tf_out)
    # print("of_out", of_out)
    test_case.assertTrue(np.allclose(tf_out, of_out))


def test_scatter_nd_update(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["params_shape"] = [(10,)]
    arg_dict["indices_shape"] = [(5, 1)]
    arg_dict["updates_shape"] = [(5,)]
    for arg in GenArgList(arg_dict):
        _compare_scatter_nd_update_with_tf(test_case, *arg)


def test_scatter_nd_update_many_dims(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["params_shape"] = [(20, 10, 10, 3, 3)]
    arg_dict["indices_shape"] = [(2, 3, 3)]
    arg_dict["updates_shape"] = [(2, 3, 3, 3)]
    for arg in GenArgList(arg_dict):
        _compare_scatter_nd_update_with_tf(test_case, *arg)


def test_scatter_nd_update_mirrored(test_case):
    params = np.random.randint(1024, size=(10,)).astype(np.float32)
    updates = np.random.randint(1024, size=(5)).astype(np.float32)
    indices = np.arange(10)
    np.random.shuffle(indices)
    indices = indices[:5].reshape(5, 1).astype(np.int32)
    _compare_scatter_nd_update_mirrored_with_tf(test_case, params, indices, updates)
