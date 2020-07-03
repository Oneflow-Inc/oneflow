from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, x_shape, axis):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    def check_grad(x_diff_blob):
        assert np.array_equal(x_diff_blob.ndarray(), np.ones(x_shape))

    @flow.global_function(func_config)
    def SqueezeJob():
        with flow.fixed_placement(device_type, "0:0"):
            x = flow.get_variable(
                "var",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.ones_initializer(),
                trainable=True,
            )
            flow.watch_diff(x, check_grad)
            loss = flow.squeeze(x, axis)
            flow.losses.add_loss(loss)
            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = SqueezeJob().get().ndarray()
    # TensorFlow
    tf_out = tf.squeeze(np.ones(x_shape, dtype=np.float32), axis).numpy()
    tf_out = np.array([tf_out]) if isinstance(tf_out, np.float32) else tf_out

    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(1, 10, 1, 10, 1)]
    arg_dict["axis"] = [None, [2], [-3], [0, 2, 4], [-1, -3, -5]]

    return GenArgList(arg_dict)


def test_squeeze(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
    compare_with_tensorflow("gpu", (1, 1, 1), [0, 1, 2])
    compare_with_tensorflow("cpu", (1, 1, 1), [0, 1, 2])
    compare_with_tensorflow("gpu", (5, 6, 7), None)
    compare_with_tensorflow("cpu", (5, 6, 7), None)
