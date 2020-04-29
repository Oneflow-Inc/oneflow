import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save

def compare_with_tensorflow(device_type, x, y, mirrored):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    #func_config.train.primary_lr(1e-4)
    #func_config.train.model_update_conf(dict(naive_conf={}))
    #with flow.device_prior_placement(device_type, "0:0"):

    # OneFlow
    if mirrored:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.function(func_config)
        def LessEqualJob(a=flow.MirroredTensorDef(x.shape),
                b=flow.MirroredTensorDef(y.shape)):
            z = flow.math.less_equal(a, b)
            return z
        of_out = LessEqualJob([x], [y]).get().ndarray_list()[0]
    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.function(func_config)
        def LessEqualJob(a=flow.FixedTensorDef(x.shape),
                b=flow.FixedTensorDef(y.shape)):
            z = flow.math.less_equal(a, b)
            return z
        of_out  = LessEqualJob(x, y).get().ndarray()

    # TensorFlow
    tf_out = tf.math.less_equal(x, y)

    assert np.allclose(of_out, tf_out.numpy(), rtol=1e-5, atol=1e-5)


def test_less_equal(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x"] = [np.array([2,3,4,8,9,12,25,36], dtype=np.float32)]
    arg_dict["y"] = [np.array([21,0,3,6,9,2,5,40], dtype=np.float32)]
    arg_dict["mirrored"] = [True]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_less_equal_1(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x"] = [np.array([6,3,4,8,9,120,2,3,6], dtype=np.float32)]
    arg_dict["y"] = [np.array([0,0,3,32,900,2,5,4,0], dtype=np.float32)]
    arg_dict["mirrored"] = [False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

