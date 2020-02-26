import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


def compare_with_tensorflow(device_type, in_shape, dir):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def SortJob(
        input=flow.MirroredTensorDef(tuple([dim + 10 for dim in in_shape]), dtype=flow.float32)
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.math.sort(input, dir)

    input = (np.random.random(in_shape) * 100).astype(np.float32)
    # OneFlow
    of_out = SortJob([input]).get().ndarray_list()[0]
    # TensorFlow
    tf_out = tf.sort(input, axis=-1, direction=dir)

    assert np.allclose(of_out, tf_out.numpy())


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(100,), (100, 100), (1000, 1000), (10, 10, 2000), (10, 100000)]
    arg_dict["dir"] = ["ASCENDING", "DESCENDING"]

    return GenArgList(arg_dict)


def test_top_k(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
