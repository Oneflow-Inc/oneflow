import oneflow as flow
import numpy as np
import os
from collections import OrderedDict 
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type

from test_util import GenArgList
import test_global_storage


def _check(test_case, x, y, shared_axes):
    alpha_of = test_global_storage.Get("alpha")
    alpha = np.expand_dims(alpha_of, axis = 0)
    dy = test_global_storage.Get("loss_diff")
    np_prelu_out = np.where(x>0, x, x*alpha)
    np_prelu_x_diff = np.where(x>0, dy, dy*alpha)
    np_prelu_alpha_diff =  np.where(x>0, 0, dy*x)
    np_prelu_alpha_diff = np.add.reduce(np_prelu_alpha_diff, axis=shared_axes, keepdims=True)
    np_prelu_alpha_diff = np.add.reduce(np_prelu_alpha_diff, axis=0)
    test_case.assertTrue(np.allclose(np_prelu_out, y))
    test_case.assertTrue(np.allclose(np_prelu_x_diff, test_global_storage.Get("x_diff")))
    test_case.assertTrue(np.allclose(np_prelu_alpha_diff, test_global_storage.Get("alpha_diff")))

def _run_test(test_case, device_type, dtype, x_shape, shared_axes):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def PreluJob(x=flow.FixedTensorDef(x_shape, dtype=type_name_to_flow_type[dtype])):
        with flow.fixed_placement(device_type, "0:0"):
            x += flow.get_variable(name='v1', shape=(1,),
                                   dtype=type_name_to_flow_type[dtype], initializer=flow.zeros_initializer())
            loss = flow.layers.prelu(
                x, alpha_initializer=flow.random_uniform_initializer(minval=0.1,maxval=0.9), shared_axes=shared_axes, name="prelu")
            alpha_shape = list(x.shape[1:])
            if shared_axes is not None:
                for i in shared_axes:
                    alpha_shape[i - 1] = 1 
            alpha = flow.get_variable(
                "prelu-alpha",
                shape=tuple(alpha_shape),
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(minval=0.1,maxval=0.9),
                )
            flow.losses.add_loss(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(alpha, test_global_storage.Setter("alpha"))
            flow.watch_diff(alpha, test_global_storage.Setter("alpha_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss
    check_point = flow.train.CheckPoint()
    check_point.init()
    x = (np.random.random(x_shape)-1).astype(type_name_to_np_type[dtype])
    y = PreluJob(x).get()
    _check(test_case, x, y.ndarray(), shared_axes)
   
def test_prelu(test_case):
    arg_dict = OrderedDict()
    arg_dict["test_case"] = [test_case]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["dtype"] = ["float32", "double"]
    arg_dict["x_shape"] = [(10, 32, 20, 20)]
    arg_dict["shared_axes"] = [(2,3), (1,), (1, 2), (1, 2, 3)]

    for arg in GenArgList(arg_dict):
        _run_test(*arg)
