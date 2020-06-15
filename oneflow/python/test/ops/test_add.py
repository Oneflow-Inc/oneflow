from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import Args, CompareOpWithTensorFlow, GenArgDict

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_naive(test_case):
    @flow.function(func_config)
    def AddJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((5, 2))):
        return a + b + b

    x = np.random.rand(5, 2).astype(np.float32)
    y = np.random.rand(5, 2).astype(np.float32)
    z = None
    z = AddJob(x, y).get().ndarray()
    test_case.assertTrue(np.array_equal(z, x + y + y))


def test_broadcast(test_case):
    @flow.function(func_config)
    def AddJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((1, 2))):
        return a + b

    x = np.random.rand(5, 2).astype(np.float32)
    y = np.random.rand(1, 2).astype(np.float32)
    z = None
    z = AddJob(x, y).get().ndarray()
    test_case.assertTrue(np.array_equal(z, x + y))


def test_xy_add_x1(test_case):
    GenerateTest(test_case, (64, 64), (64, 1))


def test_xy_add_1y(test_case):
    GenerateTest(test_case, (64, 64), (1, 64))


def test_xyz_add_x1z(test_case):
    GenerateTest(test_case, (64, 64, 64), (64, 1, 64))


def test_xyz_add_1y1(test_case):
    GenerateTest(test_case, (64, 64, 64), (1, 64, 1))


def GenerateTest(test_case, a_shape, b_shape):
    @flow.function(func_config)
    def AddJob(a=flow.FixedTensorDef(a_shape), b=flow.FixedTensorDef(b_shape)):
        return a + b

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)
    y = AddJob(a, b).get().ndarray()
    test_case.assertTrue(np.array_equal(y, a + b))


def test_scalar_add(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["flow_op"] = [flow.math.add]
    arg_dict["tf_op"] = [tf.math.add]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict["op_args"] = [
        Args([1]),
        Args([-1]),
        Args([84223.19348]),
        Args([-3284.139]),
    ]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)
