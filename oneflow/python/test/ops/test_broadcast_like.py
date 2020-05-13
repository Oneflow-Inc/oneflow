import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict
import oneflow.core.common.data_type_pb2 as data_type_util

from test_util import GenArgList


def test_broadcast_like_forward(test_case, device_type, input_shape, like_shape, axis):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    @flow.function(func_config)
    def broadcast_like_forward(x=flow.FixedTensorDef(shape=input_shape, dtype=data_type_util.kFloat), y=flow.FixedTensorDef(shape=like_shape, dtype=data_type_util.kFloat)):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.broadcast_like(x, y, axis=axis)

    x = np.random.rand(*input_shape).astype(np.float32)
    like = np.random.rand(*like_shape).astype(np.float32)
    print(like)
    of_out = broadcast_like_forward(x, like).get()
    print(of_out.ndarray())


def test_broadcast_like(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(2,4)]
    arg_dict["like_shape"] = [(2,3,4)]
    arg_dict["axis"] = [[1]]
    for arg in GenArgList(arg_dict):
        test_broadcast_like_forward(test_case, *arg)