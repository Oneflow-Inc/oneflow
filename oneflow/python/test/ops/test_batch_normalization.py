import tensorflow as tf
import oneflow as flow
import numpy as np

from test_util import GenArgDict
from test_util import RunOneflowOp
from test_util import Args
from collections import OrderedDict

def test_no_watch_scope_consistent(test_case): 
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        return flow.layers.batch_normalization(x)
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def TODO_test_no_watch_scope(test_case): 
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        return flow.layers.batch_normalization(x)
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def test_train_consistent(test_case): 
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0.001)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def TODO_test_train(test_case): 
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0.001)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))

def test_watch_scope(test_case): 
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0.001)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        with flow.watch_scope({}, {}):
            y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))
    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))


def RunTensorFlowBn(x, tf_args):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        tf_op = tf.keras.layers.BatchNormalization(*tf_args)
        y = tf_op(x)
    x_diff = tape.gradient(y, x)
    return y.numpy(), x_diff.numpy()


def CompareOpWithTensorFlow(device_type, flow_op, input_shape,
                            op_args=None, input_minval=-10, input_maxval=10, y_rtol=1e-5,
                            y_atol=1e-5, x_diff_rtol=1e-5, x_diff_atol=1e-5):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval,
                          size=input_shape).astype(np.float32)
    of_y, of_x_diff = RunOneflowOp(device_type, flow_op, x, flow_args)
    tf_y, tf_x_diff = RunTensorFlowBn(x, tf_args)

    print(of_y)
    print(tf_y)

    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(
        of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol
    )


def test_batchnorm(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict['flow_op'] = [flow.layers.batch_normalization]
    arg_dict['input_shape'] = [(1,1,1,1)]
    arg_dict['op_args'] = [Args([1])]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)
