import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict 

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


def compare_with_tensorflow(device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def ReduceSumJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ReduceSumJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        tf_out = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
#    print("tf: ")
#    print(tf_out.numpy())
#    print("of: ")
#    print(of_out.ndarray())
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=rtol, atol=atol
    )

def test_reduce_sum(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(64, 64, 64)]
    arg_dict["axis"] = [None, [1], [0, 2]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_matrix_col_reduce_sum(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(64, 64)]
    arg_dict["axis"] = [[0]]
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
