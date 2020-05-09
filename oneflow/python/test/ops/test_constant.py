import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, value, shape, rtol=1e-5, atol=1e-5):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    @flow.function(func_config)
    def ConstantJob():
        with flow.device_prior_placement(device_type, "0:0"):
            loss = flow.constant(value, dtype=flow.float, shape=shape)
            return loss

    of_out = ConstantJob().get()
    tf_out = tf.constant(value, dtype=float, shape=shape)
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=rtol, atol=atol), (
        of_out.ndarray(),
        tf_out.numpy(),
    )


def test_constant(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["value"] = [6, 6.66]
    arg_dict["shape"] = [(2, 3), (3, 3, 3)]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

