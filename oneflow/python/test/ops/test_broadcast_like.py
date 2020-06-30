import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.core.common.data_type_pb2 as data_type_util
import tensorflow as tf
from test_util import GenArgList


def compare_broadcast_like_with_tf(
    device_type, input_shape, like_shape, broadcast_axes, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def broadcast_like_forward(
        x=flow.FixedTensorDef(shape=input_shape, dtype=data_type_util.kFloat),
        y=flow.FixedTensorDef(shape=like_shape, dtype=data_type_util.kFloat),
    ):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.broadcast_like(x, y, broadcast_axes=broadcast_axes)

    x = np.random.rand(*input_shape).astype(np.float32)
    like = np.random.rand(*like_shape).astype(np.float32)
    of_out = broadcast_like_forward(x, like).get()
    np_out = np.broadcast_to(x, like_shape)
    assert np.allclose(of_out.ndarray(), np_out, rtol=rtol, atol=atol)


def test_broadcast_like(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(5, 2)]
    arg_dict["like_shape"] = [(4, 5, 2)]
    arg_dict["broadcast_axes"] = [[0]]
    for arg in GenArgList(arg_dict):
        compare_broadcast_like_with_tf(*arg)


def test_broadcast_like2(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(5, 2)]
    arg_dict["like_shape"] = [(4, 6, 5, 2)]
    arg_dict["broadcast_axes"] = [[0, 1]]
    for arg in GenArgList(arg_dict):
        compare_broadcast_like_with_tf(*arg)
