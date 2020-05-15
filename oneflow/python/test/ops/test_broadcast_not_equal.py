import oneflow as flow
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _random_inputs(x_shape, y_shape):
    x = np.random.rand(*x_shape).astype(np.float32)
    y = np.random.rand(*y_shape).astype(np.float32)
    return x, y


def _make_not_equal_fn(x, y, device_type, mirrored):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    if mirrored:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    def do_not_equal(x_blob, i_blob):
        with flow.device_prior_placement(device_type, "0:0"):
            y = flow.math.not_equal(x_blob, i_blob)
        return y

    if mirrored:

        @flow.function(func_config)
        def not_equal_fn(
            x_def=flow.MirroredTensorDef(x.shape, dtype=flow.float),
            y_def=flow.MirroredTensorDef(y.shape, dtype=flow.float),
        ):
            return do_not_equal(x_def, y_def)

    else:

        @flow.function(func_config)
        def not_equal_fn(
            x_def=flow.FixedTensorDef(x.shape, dtype=flow.float),
            y_def=flow.FixedTensorDef(y.shape, dtype=flow.float),
        ):
            return do_not_equal(x_def, y_def)

    return not_equal_fn

def _compare_not_equal_with_tf(test_case, device_type, x_shape,
        y_shape, mirrored=False):
    x, y = _random_inputs(x_shape, y_shape)
    not_equal_fn = _make_not_equal_fn(x, y, device_type, mirrored)

    if mirrored:
        of_y = not_equal_fn([x], [y]).get().ndarray_list()[0]
    else:
        of_y = not_equal_fn(x, y).get().ndarray()
    tf_y = tf.math.not_equal(x, y)
    test_case.assertTrue(np.array_equal(tf_y, of_y))

def test_not_equal(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(3, 1, 4)]
    arg_dict["y_shape"] = [(3,1, 4)]

    for arg in GenArgList(arg_dict):
        _compare_not_equal_with_tf(test_case, *arg)


def test_not_equal_case_1(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(2, 10, 2)]
    arg_dict["y_shape"] = [(10, 1)]
    for arg in GenArgList(arg_dict):
        _compare_not_equal_with_tf(test_case, *arg)


def test_not_equal_case_2(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(209, 80)]
    arg_dict["y_shape"] = [(1, 80)]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        _compare_not_equal_with_tf(test_case, *arg)

def test_not_equal_case_3(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(2, 500, 20, 2)]
    arg_dict["y_shape"] = [(2, 1, 1, 2)]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        _compare_not_equal_with_tf(test_case, *arg)


