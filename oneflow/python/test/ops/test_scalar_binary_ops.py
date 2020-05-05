import oneflow as flow
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from test_util import GenArgDict
from test_util import Save
from test_util import GetSavePath
import os


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def RunOneflowScalarBinaryOp(device_type, flow_op, x, operand, op_type, test_sbp=False):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(0)
    func_config.train.model_update_conf(dict(naive_conf={}))
    if test_sbp:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
        flow.config.gpu_device_num(2)
    @flow.function(func_config)
    def FlowJob(x=flow.FixedTensorDef(x.shape)):
        with flow.device_prior_placement(device_type, "0:0"):
            x += flow.get_variable(name='v1', shape=(1,),
                                   dtype=flow.float, initializer=flow.zeros_initializer())
            if op_type in ['left', 'commutative']:
                loss = flow_op(operand, x)
            else:
                loss = flow_op(x, operand)
            flow.losses.add_loss(loss)

            if not test_sbp:
                # watch_diff doesn't work on multiple gpus
                flow.watch_diff(x, Save("x_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    y = FlowJob(x).get().ndarray()
    if test_sbp:
        x_diff = None
    else:
        x_diff = np.load(os.path.join(GetSavePath(), "x_diff.npy"))
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
                                        operand, op_type, input_minval=-10, input_maxval=10, y_rtol=1e-5,
                                        y_atol=1e-5, x_diff_rtol=1e-5, x_diff_atol=1e-5):
    assert device_type in ["gpu", "cpu"]

    x = np.random.uniform(low=input_minval, high=input_maxval,
                          size=input_shape).astype(np.float32)
    of_y, of_x_diff, = RunOneflowScalarBinaryOp(
        device_type, flow_op, x, operand, op_type)
    tf_y, tf_x_diff = RunTensorFlowScalarBinaryOp(tf_op, x, operand, op_type)

    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(
        of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol
    )


def GenerateTfComparisonTest(flow_op, tf_op, op_types):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict['flow_op'] = [flow_op]
    arg_dict['tf_op'] = [tf_op]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict['operand'] = [1, -1, 123.438512, -235328.8313, -0.2123, 0.423]
    arg_dict['op_type'] = op_types
    for arg in GenArgDict(arg_dict):
        CompareScalarBinaryOpWithTensorFlow(**arg)


def GenerateSbpTest(flow_op, op_types):
    x = np.random.uniform(low=-10, high=10, size=(10, 10, 10)).astype(np.float32)
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict['flow_op'] = [flow_op]
    arg_dict['x'] = [x]
    arg_dict['operand'] = [1, -1, 123.438512, -235328.8313, -0.2123, 0.423]
    arg_dict['op_type'] = op_types
    arg_dict['test_sbp'] = [True]
    for arg in GenArgDict(arg_dict):
        RunOneflowScalarBinaryOp(**arg)


def test_scalar_add(test_case):
    GenerateTfComparisonTest(flow.math.add, tf.math.add, ['commutative'])


def test_scalar_sub(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    GenerateTfComparisonTest(flow.math.subtract, tf.math.subtract, ['left', 'right'])


def test_scalar_mul(test_case):
    GenerateTfComparisonTest(flow.math.multiply, tf.math.multiply, ['commutative'])


def test_scalar_div(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    # the grad of left_scalar_div has not been implemented
    GenerateTfComparisonTest(flow.math.divide, tf.math.divide, ['right'])


def test_scalar_add_sbp(test_case):
    GenerateSbpTest(flow.math.add, ['commutative'])


def test_scalar_sub_sbp(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    GenerateSbpTest(flow.math.subtract, ['left', 'right'])


def test_scalar_mul_sbp(test_case):
    GenerateSbpTest(flow.math.multiply, ['commutative'])


def test_scalar_div_sbp(test_case):
    if os.getenv("ENABLE_USER_OP") != 'True':
        return
    # the grad of left_scalar_div has not been implemented
    GenerateSbpTest(flow.math.divide, ['right'])
