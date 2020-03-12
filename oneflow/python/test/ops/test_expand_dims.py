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

    def test_expand_dims_grad(x_diff_blob):
        x_diff = x_diff_blob.ndarray()
        assert np.array_equal(x_diff, np.ones(x_shape))

    @flow.function(func_config)
    def ExpandDimsJob():
        with flow.fixed_placement(device_type, "0:0"):
            x = flow.get_variable(
                shape=x_shape,
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.ones_initializer(type_name_to_flow_type[data_type]),
                name="variable",
            )
            x = flow.identity(x)
            y = flow.expand_dims(x, axis)
            flow.watch_diff(x, test_expand_dims_grad)
            return y

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ExpandDimsJob().get().ndarray()
    # TensorFlow
    tf_out = tf.expand_dims(np.ones(x_shape, dtype=type_name_to_np_type[data_type]), axis).numpy()

    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(10, 10)]
    arg_dict["axis"] = [0, 1, 2, -1, -2, -3]
    arg_dict["data_type"] = ["float32", "double", "int8", "int32", "int64"]

    return GenArgList(arg_dict)


def test_expand_dims(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
