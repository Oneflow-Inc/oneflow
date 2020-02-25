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
    def TopKJob():
        with flow.device_prior_placement(device_type, "0:0"):
            input = flow.get_variable(
                "input",
                shape=in_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=False,
            )
            flow.watch(input, Save("input"))

            return flow.math.top_k(input, k, sorted)

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = TopKJob().get()
    # TensorFlow
    input = tf.Variable(np.load(os.path.join(GetSavePath(), "input.npy")))
    _, tf_out = tf.math.top_k(input, k, sorted)

    assert np.allclose(of_out.ndarray(), tf_out.numpy())


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu"]
    arg_dict["in_shape"] = [(100,), (100, 100), (1000, 1000), (10, 10, 2000)]
    arg_dict["k"] = [1, 50, 100, 200, 256]
    arg_dict["sorted"] = [True]

    return GenArgList(arg_dict)


def test_top_k(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
