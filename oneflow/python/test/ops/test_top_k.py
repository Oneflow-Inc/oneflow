import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


def compare_with_tensorflow(device_type, in_shape, k, sorted):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def TopKJob(
        input=flow.MirroredTensorDef(tuple([dim + 10 for dim in in_shape]), dtype=flow.float32)
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.math.top_k(input, k, sorted)

    input = (np.random.random(in_shape) * 100).astype(np.float32)
    # OneFlow
    of_out = TopKJob([input]).get().ndarray_list()[0]
    # TensorFlow
    if k <= in_shape[-1]:
        _, tf_out = tf.math.top_k(tf.Variable(input), k, sorted)
    else:
        tf_out = tf.argsort(input, axis=-1, direction="DESCENDING", stable=True)

    assert np.allclose(of_out, tf_out.numpy())


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu"]
    arg_dict["in_shape"] = [(100,), (100, 100), (1000, 1000), (10, 10, 2000), (10, 1000000)]
    arg_dict["k"] = [1, 50, 100, 200, 256]
    arg_dict["sorted"] = [True]

    return GenArgList(arg_dict)


def test_top_k(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
