import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def compare_with_tensorflow(device_type, in_shape, axis, data_type):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def SqueezeJob(input=flow.MirroredTensorDef(in_shape, dtype=type_name_to_flow_type[data_type])):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.squeeze(input, axis)

    input = (np.random.random(in_shape) * 100).astype(type_name_to_np_type[data_type])
    # OneFlow
    of_out = SqueezeJob([input]).get().ndarray_list()[0]
    # TensorFlow
    tf_out = tf.squeeze(input, axis).numpy()
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
