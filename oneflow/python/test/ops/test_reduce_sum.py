import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict 

from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def ReduceSumJob(x=flow.FixedTensorDef(input_shape)):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)
    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceSumJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
#    print("tf: ")
#    print(tf_out.numpy())
#    print("of: ")
#    print(of_out.ndarray())
#    print("diff: ")
#    print(of_out.ndarray() - tf_out.numpy())
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=rtol, atol=atol), (of_out.ndarray(), tf_out.numpy())

def test_reduce_sum(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(64, 64, 64)]
    arg_dict["axis"] = [None, [1], [0, 2]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_col_reduce(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(1024 * 64, 25)]
    arg_dict["axis"] = [[0]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg, atol=1e-1)

def test_row_reduce(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(25, 1024 * 1024)]
    arg_dict["axis"] = [[1]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_scalar(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(1024 * 64, 25)]
    arg_dict["axis"] = [[0, 1]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_sum(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)
    Foo(np.ndarray((10,), dtype=np.float32))
