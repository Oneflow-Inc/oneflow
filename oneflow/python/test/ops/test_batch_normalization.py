import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import Args, GenArgDict, type_name_to_flow_type, type_name_to_np_type


def test_no_watch_scope_consistent(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        return flow.layers.batch_normalization(x)

    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))


def TODO_test_no_watch_scope(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
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

    @flow.global_function(func_config)
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

    @flow.global_function(func_config)
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

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((2, 8, 32, 32))):
        with flow.watch_scope({}, {}):
            y = flow.layers.batch_normalization(x, axis=1)
        flow.losses.add_loss(flow.math.reduce_sum(y))

    Foo(np.ones((2, 8, 32, 32), dtype=np.float32))


def CompareNnBnWithTensorFlow(
    input_shape,
    data_type,
    axis,
    epsilon,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-5,
    y_atol=1e-5,
    x_diff_rtol=1e-5,
    x_diff_atol=1e-5,
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float32)
    func_config.train.primary_lr(0)
    func_config.train.model_update_conf(dict(naive_conf={}))

    x = np.random.uniform(low=input_minval, high=input_maxval, size=input_shape).astype(
        np.float32
    )
    param_shape = input_shape[axis]
    mean = np.random.uniform(
        low=input_minval, high=input_maxval, size=param_shape
    ).astype(np.float32)
    variance = np.random.uniform(low=0, high=input_maxval, size=param_shape).astype(
        np.float32
    )
    offset = np.random.uniform(
        low=input_minval, high=input_maxval, size=param_shape
    ).astype(np.float32)
    scale = np.random.uniform(
        low=input_minval, high=input_maxval, size=param_shape
    ).astype(np.float32)

    @flow.global_function(func_config)
    def FlowNnBnJob(
        x_full_precision=flow.FixedTensorDef(x.shape),
        mean=flow.FixedTensorDef(mean.shape),
        variance=flow.FixedTensorDef(variance.shape),
        offset=flow.FixedTensorDef(offset.shape),
        scale=flow.FixedTensorDef(scale.shape),
    ):
        with flow.device_prior_placement("gpu", "0:0"):
            x_full_precision += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
            )
            if data_type == "float16":
                x = flow.cast(x_full_precision, flow.float16)
            else:
                x = x_full_precision
            y = flow.nn.batch_normalization(
                x, mean, variance, offset, scale, epsilon, axis=axis
            )
            y = flow.cast(y, flow.float32)
            flow.losses.add_loss(y)
            flow.watch_diff(x_full_precision, test_global_storage.Setter("x_diff"))
            return y

    check_point = flow.train.CheckPoint()
    check_point.init()
    of_y = FlowNnBnJob(x, mean, variance, offset, scale).get().ndarray()
    of_x_diff = test_global_storage.Get("x_diff")

    def TensorFlowNnBn(x, mean, variance, offset, scale):
        tf_params_shape = [1, 1, 1, 1]
        tf_params_shape[axis] = input_shape[axis]
        with tf.GradientTape(persistent=True) as tape:
            x = tf.Variable(x)
            if data_type == "float16":
                x = tf.cast(x, tf.float16)
            mean = tf.Variable(mean.reshape(tf_params_shape))
            variance = tf.Variable(variance.reshape(tf_params_shape))
            offset = tf.Variable(offset.reshape(tf_params_shape))
            scale = tf.Variable(scale.reshape(tf_params_shape))
            y = tf.cast(
                tf.nn.batch_normalization(x, mean, variance, offset, scale, epsilon),
                tf.float32,
            )
        x_diff = tape.gradient(y, x)
        return y.numpy(), x_diff.numpy()

    tf_y, tf_x_diff = TensorFlowNnBn(x, mean, variance, offset, scale)
    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol)


def RunTensorFlowBn(x, tf_args, training, trainable):
    x = x.astype(np.float32)
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


def RunOneflowLayerBn(
    device_type, x, data_type, flow_args, training=True, trainable=True
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    if data_type == "float16":
        func_config.enable_auto_mixed_precision(True)
        dtype = flow.float
        np_dtype = np.float32
    else:
        dtype = type_name_to_flow_type[data_type]
        np_dtype = type_name_to_np_type[data_type]
    x = x.astype(np_dtype)

    func_config.default_data_type(dtype)
    if trainable:
        func_config.train.primary_lr(0)
        func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def FlowJob(x_full_precision=flow.FixedTensorDef(x.shape, dtype=dtype)):
        with flow.device_prior_placement(device_type, "0:0"):
            x_full_precision += flow.get_variable(
                name="v1", shape=(1,), dtype=dtype, initializer=flow.zeros_initializer()
            )
            if data_type == "float16":
                x = flow.cast(x_full_precision, flow.float16)
            else:
                x = x_full_precision
            y = flow.layers.batch_normalization(
                x, *flow_args, trainable=trainable, training=training
            )
            y = flow.cast(y, flow.float)
            if trainable:
                flow.losses.add_loss(y)

                flow.watch_diff(x_full_precision, test_global_storage.Setter("x_diff"))

            return y

    check_point = flow.train.CheckPoint()
    check_point.init()
    y = FlowJob(x).get().ndarray()
    if trainable:
        x_diff = test_global_storage.Get("x_diff")
        return y, x_diff
    else:
        return y


def CompareFp16WithFp32(
    device_type,
    input_shape,
    op_args=None,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-5,
    y_atol=1e-5,
    x_diff_rtol=1e-5,
    x_diff_atol=1e-5,
    training=True,
    trainable=True,
):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval, size=input_shape)
    if trainable:
        y_fp16, x_diff_fp16 = RunOneflowLayerBn(
            device_type, x, "float16", flow_args, training=training, trainable=trainable
        )
        y_fp32, x_diff_fp32 = RunOneflowLayerBn(
            device_type, x, "float32", flow_args, training=training, trainable=trainable
        )
        assert np.allclose(y_fp16, y_fp32, rtol=y_rtol, atol=y_atol)
        assert np.allclose(x_diff_fp16, x_diff_fp32, rtol=x_diff_rtol, atol=x_diff_atol)
    else:
        y_fp16 = RunOneflowLayerBn(
            device_type, x, "float16", flow_args, training=training, trainable=trainable
        )
        y_fp32 = RunOneflowLayerBn(
            device_type, x, "float32", flow_args, training=training, trainable=trainable
        )
        assert np.allclose(y_fp16, y_fp32, rtol=y_rtol, atol=y_atol)


def CompareBnWithTensorFlow(
    device_type,
    input_shape,
    data_type,
    op_args=None,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-5,
    y_atol=1e-5,
    x_diff_rtol=1e-5,
    x_diff_atol=1e-5,
    training=True,
    trainable=True,
):
    assert device_type in ["gpu", "cpu"]
    # tf bn doesn't support double
    assert data_type in ["float32"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval, size=input_shape)
    if trainable:
        of_y, of_x_diff = RunOneflowLayerBn(
            device_type, x, data_type, flow_args, training=training, trainable=trainable
        )
        tf_y, tf_x_diff = RunTensorFlowBn(
            x, tf_args, training=training, trainable=trainable
        )
        assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
        assert np.allclose(of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol)
    else:
        of_y = RunOneflowLayerBn(
            device_type, x, data_type, flow_args, training=training, trainable=trainable
        )
        tf_y = RunTensorFlowBn(x, tf_args, training=training, trainable=trainable)
        assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)


def test_layer_batchnorm(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["data_type"] = ["float32"]
    arg_dict["input_shape"] = [(1, 4, 1, 2)]
    arg_dict["op_args"] = [
        Args([1]),
        Args([2]),
        Args([1, 0.95, 0.0001]),
        Args([1, 0.99, 0.001, False]),
        Args([1, 0.99, 0.001, False, False]),
        Args([]),
        Args([1, 0.95, 0.1]),
    ]
    for arg in GenArgDict(arg_dict):
        CompareBnWithTensorFlow(**arg)


def test_layer_batchnorm_inference(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["data_type"] = ["float32"]
    arg_dict["input_shape"] = [(1, 4, 1, 2)]
    arg_dict["op_args"] = [
        Args([1]),
        Args([2]),
        Args([1, 0.95, 0.0001]),
        Args([1, 0.99, 0.001, False]),
        Args([1, 0.99, 0.001, False, False]),
        Args([]),
        Args([1, 0.95, 0.1]),
    ]
    for arg in GenArgDict(arg_dict):
        CompareBnWithTensorFlow(**arg, training=False, trainable=False)


def test_layer_batchnorm_trainable_without_training(test_case):
    if os.getenv("ENABLE_USER_OP") != "True":
        return
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["data_type"] = ["float32"]
    arg_dict["input_shape"] = [(2, 4, 3, 5)]
    arg_dict["op_args"] = [
        Args([1]),
        Args([2]),
        Args([1, 0.95, 0.0001]),
        Args([1, 0.99, 0.001, False]),
        Args([1, 0.99, 0.001, False, False]),
        Args([]),
        Args([1, 0.95, 0.1]),
    ]
    for arg in GenArgDict(arg_dict):
        CompareBnWithTensorFlow(**arg, training=False, trainable=True)


def test_nn_batchnorm(test_case):
    if os.getenv("ENABLE_USER_OP") != "True":
        return
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [(2, 4, 3, 5)]
    arg_dict["data_type"] = ["float32"]
    arg_dict["axis"] = [1, -1]
    arg_dict["epsilon"] = [1e-5, 1e-4]
    for arg in GenArgDict(arg_dict):
        CompareNnBnWithTensorFlow(**arg)


def test_batchnorm_fp16(test_case):
    if os.getenv("ENABLE_USER_OP") != "True":
        return
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(2, 4, 3, 5)]
    arg_dict["input_minval"] = [-2]
    arg_dict["input_maxval"] = [2]
    arg_dict["op_args"] = [
        Args([1]),
        Args([2]),
        Args([1, 0.95, 0.0001]),
        Args([1, 0.99, 0.001, False]),
        Args([1, 0.99, 0.001, False, False]),
        Args([]),
        Args([1, 0.95, 0.1]),
    ]
    for arg in GenArgDict(arg_dict):
        CompareFp16WithFp32(
            **arg, training=False, trainable=False, y_rtol=1e-3, y_atol=1e-3
        )
        CompareFp16WithFp32(
            **arg,
            training=True,
            trainable=True,
            y_rtol=1e-3,
            y_atol=1e-3,
            x_diff_rtol=1e-3,
            x_diff_atol=1e-3
        )
        CompareFp16WithFp32(
            **arg,
            training=False,
            trainable=True,
            y_rtol=1e-3,
            y_atol=1e-3,
            x_diff_rtol=1e-3,
            x_diff_atol=1e-3
        )
