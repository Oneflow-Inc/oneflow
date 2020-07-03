import os
import shutil
from collections import OrderedDict

import numpy as np
import oneflow as flow
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type


def of_run(device_type, x_shape, data_type, rate, seed):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if data_type == "float16":
        func_config.enable_auto_mixed_precision(True)
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def DropoutJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=1, maxval=10),
                trainable=True,
            )
            of_out = flow.nn.dropout(x, rate=rate, seed=seed)
            loss = flow.math.square(of_out)
            flow.losses.add_loss(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(of_out, test_global_storage.Setter("out"))
            flow.watch_diff(of_out, test_global_storage.Setter("out_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = DropoutJob().get()

    of_out = test_global_storage.Get("out")
    out_diff = test_global_storage.Get("out_diff")
    assert np.allclose(
        [1 - np.count_nonzero(of_out) / of_out.size], [rate], atol=rate / 5
    )
    x = test_global_storage.Get("x")
    x_diff = test_global_storage.Get("x_diff")
    out_scale = of_out[np.where(of_out != 0)] / x[np.where(of_out != 0)]
    diff_scale = x_diff[np.where(of_out != 0)] / out_diff[np.where(of_out != 0)]
    assert np.allclose(out_scale, 1.0 / (1.0 - rate), atol=1e-5)
    assert np.allclose(diff_scale, 1.0 / (1.0 - rate), atol=1e-5)


def test_dropout(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(100, 100, 10, 20)]
    arg_dict["data_type"] = ["float32", "double", "float16"]
    arg_dict["rate"] = [0.75]
    arg_dict["seed"] = [12345, None]

    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[2] == "float16":
            continue
        of_run(*arg)
