import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def compare_with_tensorflow(device_type, in_shape, direction, data_type):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def SortJob(
        input=flow.MirroredTensorDef(
            tuple([dim + 10 for dim in in_shape]), dtype=type_name_to_flow_type[data_type]
        )
    ):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.sort(input, direction)

    input = (np.random.random(in_shape) * 100).astype(type_name_to_np_type[data_type])
    # OneFlow
    of_out = SortJob([input]).get().ndarray_list()[0]
    # TensorFlow
    tf_out = tf.sort(input, axis=-1, direction=direction)

    assert np.array_equal(of_out, tf_out.numpy())


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(100,), (100, 100), (1000, 1000), (10, 10, 2000), (10, 10000)]
    arg_dict["direction"] = ["ASCENDING", "DESCENDING"]
    arg_dict["data_type"] = ["float32", "double", "int32", "int64"]

    return GenArgList(arg_dict)


def test_sort(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
