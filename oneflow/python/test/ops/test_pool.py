import oneflow as flow
import numpy as np
from collections import OrderedDict
from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
assert len(gpus) > 0, "No GPU found"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.debugging.set_log_device_placement(True)

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
]


def test_pool(_):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["pool_conf"] = pool_confs
    arg_dict["data_type"] = ["float32"]
    arg_dict["pooling_type"] = ["AVG", "MAX"]

    for case in GenArgList(arg_dict):
        (device_type, pool_conf, data_type, pooling_type,) = case
        x_shape = pool_conf["x_shape"]
        ksize = pool_conf["ksize"]
        strides = pool_conf["strides"]
        padding = pool_conf["padding"]
        data_format = pool_conf["data_format"]
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.train.primary_lr(1e-4)
        func_config.train.model_update_conf(dict(naive_conf={}))

        # Random inputs
        x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])

        # TF results
        with tf.GradientTape(persistent=True) as tape:
            x_tf = tf.Variable(x)
            # pooling_f = None
            # if pooling_type == "AVG":
            #     pooling_f = tf.nn.avg_pool2d
            # elif pooling_type == "MAX":
            #     pooling_f = tf.nn.max_pool2d
            # else:
            #     raise ValueError("pooling_type must be AVG or MAX")
            # y_tf = pooling_f(
            #     x_tf, ksize, strides, padding, data_format=data_format
            # )
            window_shape = [ksize, ksize]
            strides = [strides, strides]
            y_tf = tf.nn.pool(
                x_tf, window_shape, pooling_type, strides=strides, padding=padding,
                data_format=data_format
            )


        dx_tf = tape.gradient(y_tf, x_tf, tf.constant(1.0, shape=y_tf.shape))

        def assert_grad(b):
            assert np.allclose(dx_tf.numpy(), b.ndarray()), (
                case,
                dx_tf.numpy(),
                b.ndarray(),
            )

        # 1F results
        dtype = type_name_to_flow_type[data_type]

        @flow.function(func_config)
        def pooling_job(x=flow.FixedTensorDef(x_shape, dtype=dtype)):
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
                    pooling_f = flow.nn.avg_pool2d
                elif pooling_type == "MAX":
                    pooling_f = flow.nn.max_pool2d
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

        y = pooling_job(x).get()
        assert y.ndarray().shape == y_tf.numpy().shape, (y.ndarray().shape, y_tf.numpy().shape)
        assert np.allclose(y.ndarray(), y_tf.numpy(), rtol=1e-5, atol=1e-5), (case, y.ndarray() - y_tf.numpy())
