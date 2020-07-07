import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def WatchDiff(enable_eager_execution, test_case, device_type, input_shape, dtype):
    assert device_type in ["gpu", "cpu"]
    assert dtype in ["float32", "double"]
    flow.enable_eager_execution(enable_eager_execution)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    def CheckOnes(diff):
        ones = np.ones(input_shape)
        test_case.assertTrue(np.allclose(diff.ndarray(), ones, rtol=1e-5, atol=1e-5))

    @flow.global_function(func_config)
    def TrainJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(),
                trainable=True,
            )
            flow.watch_diff(x, CheckOnes)
            flow.losses.add_loss(x)

    check_point = flow.train.CheckPoint()
    check_point.init()
    TrainJob()
    flow.clear_default_session()


def test_watch_diff(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(10,)]
    arg_dict["dtype"] = ["float32"]
    for arg in GenArgList(arg_dict):
        WatchDiff(False, test_case, *arg)

def test_eager_watch_diff(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(10,)]
    arg_dict["dtype"] = ["float32"]
    for arg in GenArgList(arg_dict):
        WatchDiff(True, test_case, *arg)
