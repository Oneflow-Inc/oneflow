import oneflow as flow
import numpy as np
from collections import OrderedDict
from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type
import tensorflow as tf
import collections
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

pool_confs = [
    {
        "x_shape": (1, 1, 6, 6),
        "ksize": 1,
        "strides": 1,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 7, 7),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 7, 7, 3),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NHWC",
    },
    {
        "x_shape": (1, 5, 6, 6),
        "ksize": 3,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 7, 5, 5),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 3, 3),
        "ksize": 1,
        "strides": 1,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 1, 9, 9),
        "ksize": 2,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 9, 9, 1),
        "ksize": 2,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NHWC",
    },
    {
        "x_shape": (1, 1, 9, 9, 9),
        "ksize": 2,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NCDHW",
    },
    {
        "x_shape": (1, 7, 5, 5, 5),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCDHW",
    },
    {
        "x_shape": (1, 5, 5, 5, 7),
        "ksize": 3,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NDHWC",
    },
    {
        "x_shape": (1, 3, 3, 3, 3),
        "ksize": 1,
        "strides": 1,
        "padding": "VALID",
        "data_format": "NCDHW",
    },
]


def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]

    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )


def test_pool(_):
    arg_dict = OrderedDict()
    is_user_op = os.getenv("ENABLE_USER_OP") == "True"
    if is_user_op:
        arg_dict["device_type"] = ["gpu", "cpu"]
    else:
        arg_dict["device_type"] = ["gpu"]
    arg_dict["pool_conf"] = pool_confs
    arg_dict["data_type"] = ["float32"]
    arg_dict["pooling_type"] = ["AVG", "MAX"]
    arg_dict["is_dynamic"] = [True, False]

    for case in GenArgList(arg_dict):
        (device_type, pool_conf, data_type, pooling_type, is_dynamic) = case
        x_shape = pool_conf["x_shape"]
        ksize = pool_conf["ksize"]
        strides = pool_conf["strides"]
        padding = pool_conf["padding"]
        data_format = pool_conf["data_format"]
        flow.clear_default_session()

        # Random inputs
        x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])
        dim = len(x.shape) - 2

        # TODO: these cases will fail in old implementation
        if dim == 3 and data_format == "NDHWC":
            continue
        # TF results
        with tf.GradientTape(persistent=True) as tape:
            x_tf = tf.Variable(x)
            strides = _GetSequence(strides, dim, "strides")
            pooling_f = None
            if pooling_type == "AVG":
                pooling_f = getattr(tf.nn, "avg_pool{}d".format(dim))
            elif pooling_type == "MAX":
                pooling_f = getattr(tf.nn, "max_pool{}d".format(dim))
            else:
                raise ValueError("pooling_type must be AVG or MAX")
            y_tf = pooling_f(x_tf, ksize, strides, padding, data_format=data_format)

        dx_tf = tape.gradient(y_tf, x_tf, tf.constant(1.0, shape=y_tf.shape))

        def assert_grad(b):
            assert np.allclose(dx_tf.numpy(), b.ndarray()), (
                case,
                dx_tf.numpy(),
                b.ndarray(),
            )

        # 1F results
        dtype = type_name_to_flow_type[data_type]

        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.train.primary_lr(1e-4)
        func_config.train.model_update_conf(dict(naive_conf={}))

        tensor_def = None
        if is_dynamic:
            tensor_def = flow.MirroredTensorDef
        else:
            tensor_def = flow.FixedTensorDef

        @flow.function(func_config)
        def pooling_job(x=tensor_def(x_shape, dtype=dtype)):
            v = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.constant_initializer(0),
                trainable=True,
            )
            flow.watch_diff(v, assert_grad)
            x += v
            with flow.device_prior_placement(device_type, "0:0"):
                pooling_f = None
                if pooling_type == "AVG":
                    pooling_f = getattr(flow.nn, "avg_pool{}d".format(dim))
                elif pooling_type == "MAX":
                    pooling_f = getattr(flow.nn, "max_pool{}d".format(dim))
                else:
                    raise ValueError("pooling_type must be AVG or MAX")
                y = pooling_f(
                    x,
                    ksize=ksize,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                )
            flow.losses.add_loss(y)
            return y

        if is_dynamic:
            x = [x]
        y = pooling_job(x).get()
        y_ndarray = None
        if hasattr(y, "ndarray"):
            y_ndarray = y.ndarray()
        else:
            y_ndarray = y.ndarray_list()[0]
        assert y_ndarray.shape == y_tf.numpy().shape, (
            y_ndarray.shape,
            y_tf.numpy().shape,
        )
        assert np.allclose(y_ndarray, y_tf.numpy(), rtol=1e-5, atol=1e-5), (
            case,
            y_ndarray - y_tf.numpy(),
        )
