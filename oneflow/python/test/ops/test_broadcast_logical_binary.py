import oneflow as flow
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

func_config = flow.FunctionConfig()

def _random_inputs(x_shape, y_shape, data_type):
    if data_type == flow.int8:
        x = np.random.randint(0, 127, x_shape).astype(np.int8)
        y = np.random.randint(0, 127, y_shape).astype(np.int8)
    elif data_type == flow.int32:
        x = np.random.randint(0, 10000, x_shape).astype(np.int32)
        y = np.random.randint(0, 10000, y_shape).astype(np.int32)
    elif data_type == flow.int64:
        x = np.random.randint(0, 10000, x_shape).astype(np.int64)
        y = np.random.randint(0, 10000, y_shape).astype(np.int64)
    elif data_type == flow.float:
        x = np.random.rand(*x_shape).astype(np.float32)
        y = np.random.rand(*y_shape).astype(np.float32)
    elif data_type == flow.double:
        x = np.random.rand(*x_shape).astype(np.double)
        y = np.random.rand(*y_shape).astype(np.double)
    else:
        assert False
    return x, y

def of_equal(x_blob, i_blob):
    return flow.math.equal(x_blob, i_blob)

def of_not_equal(x_blob, i_blob):
    return flow.math.not_equal(x_blob, i_blob)

def of_greater_equal(x_blob, i_blob):
    y = flow.math.greater_equal(x_blob, i_blob)
    return y

def of_less_equal(x_blob, i_blob):
    y = flow.math.less_equal(x_blob, i_blob)
    return y

def of_greater(x_blob, i_blob):
    y = flow.math.greater(x_blob, i_blob)
    return y

def of_less(x_blob, i_blob):
    y = flow.math.less(x_blob, i_blob)
    return y

def of_logical_and(x_blob, i_blob):
    y = flow.math.logical_and(x_blob, i_blob)
    return y

def tf_equal(x_blob, i_blob):
    y = tf.math.equal(x_blob, i_blob)
    return y

def tf_not_equal(x_blob, i_blob):
    y = tf.math.not_equal(x_blob, i_blob)
    return y

def tf_greater_equal(x_blob, i_blob):
    y = tf.math.greater_equal(x_blob, i_blob)
    return y

def tf_greater(x_blob, i_blob):
    y = tf.math.greater(x_blob, i_blob)
    return y

def tf_less_equal(x_blob, i_blob):
    y = tf.math.less_equal(x_blob, i_blob)
    return y

def tf_less(x_blob, i_blob):
    y = tf.math.less(x_blob, i_blob)
    return y

def tf_logical_and(x_blob, i_blob):
    y = tf.math.logical_and(x_blob, i_blob)
    return y

def  _make_broadcast_logical_fn(of_func, x, y, device_type, data_type, mirrored):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    if mirrored:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    if mirrored:
        @flow.function(func_config)
        def broadcast_logical_fn(
                x_def=flow.MirroredTensorDef(x.shape, dtype=data_type),
                y_def=flow.MirroredTensorDef(y.shape, dtype=data_type),
            ):
            with flow.device_prior_placement(device_type, "0:0"):
                return eval(of_func)(x_def, y_def)

    else:
        @flow.function(func_config)
        def broadcast_logical_fn(
            x_def=flow.FixedTensorDef(x.shape, dtype=data_type),
            y_def=flow.FixedTensorDef(y.shape, dtype=data_type),
        ):
            with flow.fixed_placement(device_type, "0:0"):
                return eval(of_func)(x_def, y_def)

    return broadcast_logical_fn

def _comparebroadcast_logical_with_tf(test_case, device_type, x_shape, y_shape,
        data_type, mirrored=False):
    # flow.clear_default_session()
    x, y = _random_inputs(x_shape, y_shape, data_type)

    broadcast_logical_list = ["equal", "not_equal",
          "greater_equal", "greater",
            "less_equal", "less", "logical_and"]
    for func in broadcast_logical_list:
        of_func = "of_" + func
        broadcast_logical_func = _make_broadcast_logical_fn(of_func, x, y,
                device_type, data_type, mirrored)
        tf_func = "tf_" + func
        tf_y = eval(tf_func)(x, y)
        if mirrored:
            of_y = broadcast_logical_func([x], [y]).get().ndarray_list()[0]
            test_case.assertTrue(np.array_equal(tf_y, of_y))
            #flow.clear_default_session()

        else:
            of_y = broadcast_logical_func(x, y).get().ndarray()
            test_case.assertTrue(np.array_equal(tf_y, of_y))
           # flow.clear_default_session()


def test_broadcast_logical(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(3, 1, 4)]
    arg_dict["y_shape"] = [(3, 1, 4)]
    arg_dict["data_type"] = [flow.int32, flow.int8, flow.int32, flow.int64,
            flow.float, flow.double]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[3] == "float16": continue
        _comparebroadcast_logical_with_tf(test_case, *arg)


def testbroadcast_logical_case_1(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(2, 10, 20)]
    arg_dict["y_shape"] = [(10, 1)]
    arg_dict["data_type"] = [flow.int32, flow.int8, flow.int32, flow.int64,
            flow.float, flow.double]
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[3] == "float16": continue
        _comparebroadcast_logical_with_tf(test_case, *arg)


def testbroadcast_logical_case_2(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(209, 80)]
    arg_dict["y_shape"] = [(1, 80)]
    arg_dict["data_type"] = [flow.int32, flow.int8, flow.int32, flow.int64,
            flow.float, flow.double]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[3] == "float16": continue
        _comparebroadcast_logical_with_tf(test_case, *arg)

def testbroadcast_logical_case_3(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(20, 1, 1, 2)]
    arg_dict["y_shape"] = [(20, 500, 20, 2)]
    arg_dict["data_type"] = [flow.int32, flow.int8, flow.int32, flow.int64,
            flow.float, flow.double]
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[3] == "float16": continue
        _comparebroadcast_logical_with_tf(test_case, *arg)

def testbroadcast_logical_case_4(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [(500, 1, 20)]
    arg_dict["y_shape"] = [(20, 500, 20, 20)]
    arg_dict["data_type"] = [flow.int32, flow.int8, flow.int32, flow.int64,
            flow.float, flow.double]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[3] == "float16": continue
        _comparebroadcast_logical_with_tf(test_case, *arg)

def testbroadcast_logical_case_5(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu"]
    arg_dict["x_shape"] = [(5, 10, 20)]
    arg_dict["y_shape"] = [(10, 1)]
    arg_dict["data_type"] = [flow.int32, flow.int8, flow.int32, flow.int64,
            flow.float, flow.double]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[3] == "float16": continue
        _comparebroadcast_logical_with_tf(test_case, *arg)


