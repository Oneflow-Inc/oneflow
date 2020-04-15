import numpy as np
import math
import os
import oneflow as flow
from collections import OrderedDict 

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


of_old_activation_map = {
    "relu": flow.keras.activations.relu,
    "sigmoid": flow.keras.activations.sigmoid,
    "tanh": flow.keras.activations.tanh,
    # "gelu": flow.keras.activations.gelu,
}

of_activation_map = {
    "relu": flow.nn.relu,
    "sigmoid": flow.keras.activations.sigmoid,
    "tanh": flow.keras.activations.tanh,
    # "gelu": flow.keras.activations.gelu,
}

of_dtype2np = {
    flow.float: np.float32,
    flow.double: np.float64,
}

def compare_with_old_version_fwd(device_type, activation_type, shape, data_type):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)

    @flow.function(func_config)
    def FwdJob(x = flow.FixedTensorDef(shape, dtype=data_type)):
        with flow.device_prior_placement(device_type, "0:0"):
            return of_activation_map[activation_type](x), of_old_activation_map[activation_type](x)

    x = np.random.rand(*shape).astype(of_dtype2np[data_type]) - 0.5
    print(x)
    of_out = FwdJob(x).get()
    print(of_out[0].ndarray())
    print(of_out[1].ndarray())
    assert np.allclose(of_out[0].ndarray(), of_out[1].ndarray())

def compare_with_old_version(device_type, activation_type, shape, data_type):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def ActivationJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=data_type,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            x_old = x
            x_new = x
            loss = of_activation_map[activation_type](x_new)
            flow.losses.add_loss(loss)

            flow.watch(x_new, Save("x"))
            flow.watch_diff(x_new, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            # loss_old = of_old_activation_map[activation_type](x_old)
            # flow.losses.add_loss(loss_old)

            # flow.watch_diff(x_old, Save("x_old_diff"))
            # flow.watch(loss_old, Save("loss_old"))
            # flow.watch_diff(loss_old, Save("loss_old_diff"))

            return loss, loss#_old

    of_out = ActivationJob().get()
    print(of_out[0].ndarray())
    print(of_out[1].ndarray())
    print(GetSavePath())
    assert np.allclose(of_out[0].ndarray(), of_out[1].ndarray())

def test_activations(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["activation_type"] = ["relu"]#, "sigmoid", "tanh"]
    #arg_dict["shape"] = [(1024, 1024)]
    arg_dict["shape"] = [(4, 3)]
    arg_dict["data_type"] = [flow.float, flow.double]
    for arg in GenArgList(arg_dict):
        print(arg)
        compare_with_old_version_fwd(*arg)
