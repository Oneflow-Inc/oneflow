import oneflow as flow
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import os

from test_util import GenArgDict
import test_global_storage


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def RunOneflowScalarBinaryOp(device_type, flow_op, x, operand, op_type, multiple_gpus, only_forward):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    assert only_forward or not multiple_gpus
    if multiple_gpus:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
        flow.config.gpu_device_num(2)
        flow_env = lambda: flow.fixed_placement(device_type, "0:0-1")
    else:
        flow_env = lambda: flow.device_prior_placement(device_type, "0:0")

    if not only_forward:
        # enable training
        func_config.train.primary_lr(0)
        func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def FlowJob(x=flow.FixedTensorDef(x.shape)):
        with flow_env():
            x += flow.get_variable(name='v1', shape=(1,),
                                   dtype=flow.float, initializer=flow.zeros_initializer())
            if op_type in ['left', 'commutative']:
                y = flow_op(operand, x)
            else:
                y = flow_op(x, operand)

            if not only_forward:
                flow.losses.add_loss(y)
                flow.watch_diff(x, test_global_storage.Setter("x_diff"))

            return y

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    y = FlowJob(x).get().ndarray()
    if only_forward:
        return y, None
    else:
        x_diff = test_global_storage.Get("x_diff")
        return y, x_diff


def RunTensorFlowScalarBinaryOp(tf_op, x, operand, op_type):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        if op_type in ['left', 'commutative']:
            y = tf_op(operand, x)
        else:
            y = tf_op(x, operand)
    x_diff = tape.gradient(y, x)
    return y.numpy(), x_diff.numpy()


def CompareScalarBinaryOpWithTensorFlow(device_type, flow_op, tf_op, input_shape,
                                        operand, op_type, multiple_gpus, only_forward, input_minval=-10, input_maxval=10, y_rtol=1e-5,
                                        y_atol=1e-5, x_diff_rtol=1e-5, x_diff_atol=1e-5):
    assert device_type in ["gpu", "cpu"]

    x = np.random.uniform(low=input_minval, high=input_maxval,
                          size=input_shape).astype(np.float32)
    of_y, of_x_diff = RunOneflowScalarBinaryOp(
        device_type, flow_op, x, operand, op_type, multiple_gpus, only_forward)
    tf_y, tf_x_diff = RunTensorFlowScalarBinaryOp(tf_op, x, operand, op_type)

    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    if not only_forward:
        assert np.allclose(
            of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol
        )


def GenerateSingleGpuTest(flow_op, tf_op, op_types):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict['flow_op'] = [flow_op]
    arg_dict['tf_op'] = [tf_op]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict['operand'] = [1, -1, 123.438512, -235328.8313, -0.2123, 0.423]
    arg_dict['op_type'] = op_types
    arg_dict['multiple_gpus'] = [False]
    arg_dict['only_forward'] = [False]
    for arg in GenArgDict(arg_dict):
        CompareScalarBinaryOpWithTensorFlow(**arg)


def GenerateMultipleGpusForwardTest(flow_op, tf_op, op_types):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict['flow_op'] = [flow_op]
    arg_dict['tf_op'] = [tf_op]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict['operand'] = [1, -1, 123.438512, -235328.8313, -0.2123, 0.423]
    arg_dict['op_type'] = op_types
    arg_dict['multiple_gpus'] = [True]
    arg_dict['only_forward'] = [True]
    for arg in GenArgDict(arg_dict):
        CompareScalarBinaryOpWithTensorFlow(**arg)


def test_scalar_add_single_gpu(test_case):
    GenerateSingleGpuTest(flow.math.add, tf.math.add, ['commutative'])


def test_scalar_sub_single_gpu(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    GenerateSingleGpuTest(flow.math.subtract, tf.math.subtract, ['left', 'right'])


def test_scalar_mul_single_gpu(test_case):
    GenerateSingleGpuTest(flow.math.multiply, tf.math.multiply, ['commutative'])


def test_scalar_div_single_gpu(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    # the grad of left_scalar_div has not been implemented
    GenerateSingleGpuTest(flow.math.divide, tf.math.divide, ['right'])


def test_scalar_add_multiple_gpus_forward(test_case):
    GenerateMultipleGpusForwardTest(flow.math.add, tf.math.add, ['commutative'])


def test_scalar_sub_multiple_gpus_forward(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    GenerateMultipleGpusForwardTest(flow.math.subtract, tf.math.subtract, ['left', 'right'])


def test_scalar_mul_multiple_gpus_forward(test_case):
    GenerateMultipleGpusForwardTest(flow.math.multiply, tf.math.multiply, ['commutative'])


def test_scalar_div_multiple_gpus_forward(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    GenerateMultipleGpusForwardTest(flow.math.divide, tf.math.divide, ['left', 'right'])
