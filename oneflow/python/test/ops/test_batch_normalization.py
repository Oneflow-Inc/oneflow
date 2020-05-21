import tensorflow as tf
import oneflow as flow
import numpy as np

from test_util import GenArgDict
from test_util import Args
import test_global_storage

from collections import OrderedDict
import os

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


def RunTensorFlowBn(x, tf_args, training, trainable):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        tf_op = tf.keras.layers.BatchNormalization(*tf_args, trainable=trainable)
        y = tf_op(x, training=training)
    if trainable:
        x_diff = tape.gradient(y, x)
        return y.numpy(), x_diff.numpy()
    else:
        return y.numpy()


def RunOneflowBn(device_type, x, flow_args, training=True, trainable=True):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float)
    if trainable:
        func_config.train.primary_lr(0)
        func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def FlowJob(x=flow.FixedTensorDef(x.shape)):
        with flow.device_prior_placement(device_type, "0:0"):
            x += flow.get_variable(name='v1', shape=(1,),
                                   dtype=flow.float, initializer=flow.zeros_initializer())
            loss = flow.layers.batch_normalization(x, *flow_args, trainable=trainable, training=training)
            if trainable:
                flow.losses.add_loss(loss)

                flow.watch_diff(x, test_global_storage.Setter("x_diff"))

            return loss

    check_point = flow.train.CheckPoint()
    check_point.init()
    y = FlowJob(x).get().ndarray()
    if trainable:
        x_diff = test_global_storage.Get("x_diff")
        return y, x_diff
    else:
        return y


def CompareBnWithTensorFlow(device_type, input_shape,
                            op_args=None, input_minval=-10, input_maxval=10, y_rtol=1e-5,
                            y_atol=1e-5, x_diff_rtol=1e-5, x_diff_atol=1e-5, training=True, trainable=True):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval,
                          size=input_shape).astype(np.float32)
    if trainable:
        of_y, of_x_diff = RunOneflowBn(device_type, x, flow_args, training=training, trainable=trainable)
        tf_y, tf_x_diff = RunTensorFlowBn(x, tf_args, training=training, trainable=trainable)
        assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
        assert np.allclose(
            of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol
        )
    else:
        of_y = RunOneflowBn(device_type, x, flow_args, training=training, trainable=trainable)
        tf_y = RunTensorFlowBn(x, tf_args, training=training, trainable=trainable)
        assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)


def test_batchnorm(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict['input_shape'] = [(1,4,1,2)]
    arg_dict['op_args'] = [Args([1]), Args([2]), Args([1, 0.95, 0.0001]), Args([1, 0.99, 0.001, False]), Args([1, 0.99, 0.001, False, False]), Args([]), Args([1, 0.95, 0.1])]
    for arg in GenArgDict(arg_dict):
        CompareBnWithTensorFlow(**arg)


def test_batchnorm_inference(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict['input_shape'] = [(1,4,1,2)]
    arg_dict['op_args'] = [Args([1]), Args([2]), Args([1, 0.95, 0.0001]), Args([1, 0.99, 0.001, False]), Args([1, 0.99, 0.001, False, False]), Args([]), Args([1, 0.95, 0.1])]
    for arg in GenArgDict(arg_dict):
        CompareBnWithTensorFlow(**arg, training=False, trainable=False)


def test_batchnorm_trainable_without_training(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict['input_shape'] = [(2,4,3,5)]
    arg_dict['op_args'] = [Args([1]), Args([2]), Args([1, 0.95, 0.0001]), Args([1, 0.99, 0.001, False]), Args([1, 0.99, 0.001, False, False]), Args([]), Args([1, 0.95, 0.1])]
    for arg in GenArgDict(arg_dict):
        CompareBnWithTensorFlow(**arg, training=False, trainable=True)
