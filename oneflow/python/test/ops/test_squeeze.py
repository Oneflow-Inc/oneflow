import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def compare_with_tensorflow(device_type, x_shape, axis, data_type):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def test_squeeze_grad(x_diff_blob):
        x_diff = x_diff_blob.ndarray()
        assert np.array_equal(x_diff, np.ones(x_shape))

    @flow.function(func_config)
    def SqueezeJob():
        with flow.fixed_placement(device_type, "0:0"):
            x = flow.get_variable(
                shape=x_shape,
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.ones_initializer(type_name_to_flow_type[data_type]),
                name="variable",
            )
            x = flow.identity(x)
            y = flow.squeeze(x, axis)
            flow.watch_diff(x, test_squeeze_grad)
            return y

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = SqueezeJob().get().ndarray()
    # TensorFlow
    tf_out = tf.squeeze(np.ones(x_shape, dtype=type_name_to_np_type[data_type]), axis).numpy()
    tf_out = np.array([tf_out]) if isinstance(tf_out, type_name_to_np_type[data_type]) else tf_out

    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(1, 10, 1, 10, 1)]
    arg_dict["axis"] = [[2], [-3], [0, 2, 4], [-1, -3, -5]]
    arg_dict["data_type"] = ["float32", "double", "int8", "int32", "int64"]

    return GenArgList(arg_dict)


def test_squeeze(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
    compare_with_tensorflow("gpu", (1, 1, 1), [0, 1, 2], "float32")
    compare_with_tensorflow("cpu", (1, 1, 1), [0, 1, 2], "float32")
