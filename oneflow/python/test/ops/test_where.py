import numpy as np
import oneflow as flow
import tensorflow as tf

from collections import OrderedDict
from test_util import GenArgDict

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _random_input(cond_shape, x_shape, y_shape):
    condition = np.random.randint(low=0, high=2, size=cond_shape).astype(np.int32)
    x = np.random.standard_normal(x_shape).astype(np.float32)
    y = np.random.standard_normal(y_shape).astype(np.float32)
    return condition, x, y


def _of_where(
    condition,
    x,
    y,
    device_type="gpu",
    machine_device_ids="0:0",
    dynamic=False,
    dz_dx_watcher=None,
    dz_dy_watcher=None,
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    if callable(dz_dx_watcher) and callable(dz_dy_watcher):
        func_config.train.primary_lr(1e-3)
        func_config.train.model_update_conf(dict(naive_conf={}))

        def do_where(condition, x, y):
            with flow.device_prior_placement(device_type, "0:0"):
                x_var = flow.get_variable(
                    "x",
                    shape=x.shape,
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                x_var = x_var + x
                y_var = flow.get_variable(
                    "y",
                    shape=y.shape,
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                )
                y_var = y_var + y

            z = flow.where(condition, x_var, y_var)

            with flow.device_prior_placement(device_type, "0:0"):
                flow.losses.add_loss(z)

            flow.watch_diff(x_var, dz_dx_watcher)
            flow.watch_diff(y_var, dz_dy_watcher)
            return z

    else:

        def do_where(condition, x, y):
            return flow.where(condition, x, y)

    if dynamic:
        func_config.default_placement_scope(
            flow.device_prior_placement(device_type, "0:0")
        )
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.global_function(func_config)
        def where_fn(
            condition_def=flow.MirroredTensorDef(condition.shape, dtype=flow.int32),
            x_def=flow.MirroredTensorDef(x.shape, dtype=flow.float),
            y_def=flow.MirroredTensorDef(y.shape, dtype=flow.float),
        ):
            return do_where(condition_def, x_def, y_def)

        check_point = flow.train.CheckPoint()
        check_point.init()
        return where_fn([condition], [x], [y]).get().ndarray_list()[0]

    else:
        func_config.default_placement_scope(
            flow.device_prior_placement(device_type, machine_device_ids)
        )
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.global_function(func_config)
        def where_fn(
            condition_def=flow.FixedTensorDef(condition.shape, dtype=flow.int32),
            x_def=flow.FixedTensorDef(x.shape, dtype=flow.float),
            y_def=flow.FixedTensorDef(y.shape, dtype=flow.float),
        ):
            return do_where(condition_def, x_def, y_def)

        check_point = flow.train.CheckPoint()
        check_point.init()
        return where_fn(condition, x, y).get().ndarray()


def _compare_with_np(test_case, cond_shape, x_shape, y_shape, device_type, dynamic):
    condition, x, y = _random_input(cond_shape, x_shape, y_shape)
    z = np.where(condition, x, y)
    of_z = _of_where(condition, x, y, device_type, "0:0", dynamic)
    test_case.assertTrue(np.array_equal(z, of_z))


def _compare_with_tf(
    test_case,
    cond_shape,
    x_shape,
    y_shape,
    device_type="gpu",
    machine_device_ids="0:0",
    dynamic=False,
    verbose=False,
):
    condition, x, y = _random_input(cond_shape, x_shape, y_shape)

    condition_constant = tf.constant(condition, dtype=tf.bool)
    with tf.GradientTape(persistent=True) as t:
        x_var = tf.Variable(x)
        y_var = tf.Variable(y)
        z = tf.where(condition_constant, x_var, y_var)

    dz_dx = t.gradient(z, x_var)
    dz_dy = t.gradient(z, y_var)

    def compare_dz_dx(dz_dx_blob):
        if verbose:
            print("condition:", condition)
            print("tf_dz_dx:", dz_dx.numpy())
            print(
                "of_dz_dx:",
                dz_dx_blob.ndarray_list()[0] if dynamic else dz_dx_blob.ndarray(),
            )

        test_case.assertTrue(
            np.array_equal(
                dz_dx.numpy(),
                dz_dx_blob.ndarray_list()[0] if dynamic else dz_dx_blob.ndarray(),
            )
        )

    def compare_dz_dy(dz_dy_blob):
        if verbose:
            print("condition:", condition)
            print("tf_dz_dy:", dz_dy.numpy())
            print(
                "of_dz_dy:",
                dz_dy_blob.ndarray_list()[0] if dynamic else dz_dy_blob.ndarray(),
            )

        test_case.assertTrue(
            np.array_equal(
                dz_dy.numpy(),
                dz_dy_blob.ndarray_list()[0] if dynamic else dz_dy_blob.ndarray(),
            )
        )

    of_z = _of_where(
        condition,
        x,
        y,
        device_type,
        machine_device_ids,
        dynamic,
        compare_dz_dx,
        compare_dz_dy,
    )
    test_case.assertTrue(np.array_equal(z.numpy(), of_z))


def _of_where_with_x_and_y_are_none(input, input_shape=None):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    if input_shape is None:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.global_function(func_config)
        def where_fn(input_def=flow.FixedTensorDef(input.shape, dtype=flow.float)):
            return flow.where(input_def)

    else:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.global_function(func_config)
        def where_fn(input_def=flow.MirroredTensorDef(input_shape, dtype=flow.float)):
            return flow.where(input_def)

    return where_fn([input]).get().ndarray_list()[0]


def test_where(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[5, 10]]
    arg_dict["x_shape"] = [[5, 10]]
    arg_dict["y_shape"] = [[5, 10]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_np(test_case, **arg)


def test_where_case_1(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[4, 5, 8]]
    arg_dict["x_shape"] = [[1, 5, 8]]
    arg_dict["y_shape"] = [[4, 1, 8]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_np(test_case, **arg)


def test_where_case_2(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[10, 7, 9]]
    arg_dict["x_shape"] = [[20, 10, 7, 9]]
    arg_dict["y_shape"] = [[20, 10, 1, 1]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_np(test_case, **arg)


def test_where_case_3(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[12, 25, 6]]
    arg_dict["x_shape"] = [[12, 1, 6]]
    arg_dict["y_shape"] = [[25, 1]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_np(test_case, **arg)


def test_where_grad(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[10]]
    arg_dict["x_shape"] = [[10]]
    arg_dict["y_shape"] = [[10]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["machine_device_ids"] = ["0:0"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_tf(test_case, **arg)


def test_where_grad_case_1(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[3, 7, 10]]
    arg_dict["x_shape"] = [[3, 1, 10]]
    arg_dict["y_shape"] = [[7, 10]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_tf(test_case, **arg)


def test_where_grad_case_2(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[16, 1]]
    arg_dict["x_shape"] = [[4, 1, 20]]
    arg_dict["y_shape"] = [[8, 4, 16, 20]]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgDict(arg_dict):
        _compare_with_tf(test_case, **arg)


def test_where_grad_4card(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[10]]
    arg_dict["x_shape"] = [[10]]
    arg_dict["y_shape"] = [[10]]
    arg_dict["device_type"] = ["gpu"]
    arg_dict["machine_device_ids"] = ["0:0-3"]
    arg_dict["dynamic"] = [False]
    for arg in GenArgDict(arg_dict):
        _compare_with_tf(test_case, **arg)


def test_where_argwhere(test_case):
    rand_input = np.random.random_sample((11, 3, 5)).astype(np.float32)
    rand_input[np.nonzero(rand_input < 0.5)] = 0.0
    ret = _of_where_with_x_and_y_are_none(rand_input, input_shape=(11, 3, 5))
    exp_ret = np.argwhere(rand_input)
    test_case.assertTrue(np.array_equal(exp_ret, ret))
