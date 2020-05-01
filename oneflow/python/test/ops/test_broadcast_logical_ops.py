import oneflow as flow
import numpy as np

from collections import OrderedDict
from test_util import type_name_to_flow_type
from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save

def test_naive(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    @flow.function(func_config)
    def ModJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((5, 2))):
        return a == b

    x = np.random.rand(5, 2).astype(np.float32)
    y = np.random.rand(5, 2).astype(np.float32)
    z = ModJob(x, y).get().ndarray()
    r = func_equal(x, y)
    test_case.assertTrue(np.array_equal(z, x == y))
    flow.clear_default_session()

def test_broadcast(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    @flow.function(func_config)
    def ModJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((1, 2))):
        return a == b

    x = np.random.rand(5, 2).astype(np.float32)
    y = np.random.rand(1, 2).astype(np.float32)
    z = None
    z = ModJob(x, y).get().ndarray()
    test_case.assertTrue(np.array_equal(z, x == y))
    flow.clear_default_session()

def func_equal(a, b):
    return a == b
def func_not_equal(a, b):
    return a != b
def func_greater_than(a, b):
    return a > b
def func_greater_equal(a, b):
    return a >= b
def func_less_than(a, b):
    return a < b
def func_less_equal(a, b):
    return a <= b
#def func_logical_and(a, b):
#    return a & b

def np_array(dtype, shape):
    if dtype == flow.int8:
        return np.random.randint(0, 127, shape).astype(np.int8)
    elif dtype == flow.int32:
        return np.random.randint(0, 10000, shape).astype(np.int32)
    elif dtype == flow.int64:
        return np.random.randint(0, 10000, shape).astype(np.int64)
    elif dtype == flow.float:
        return np.random.rand(*shape).astype(np.float32)
    elif dtype == flow.double:
        return np.random.rand(*shape).astype(np.double)
    else:
        assert False


def GenerateTest(test_case, func, a_shape, b_shape, dtype=flow.int32, device_type="cpu"):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)

    @flow.function(func_config)
    def ModJob1(a=flow.FixedTensorDef(a_shape, dtype=dtype)):
        with flow.device_prior_placement(device_type, "0:0"):
            return func(a, a)

    @flow.function(func_config)
    def ModJob2(a=flow.FixedTensorDef(a_shape, dtype=dtype),
               b=flow.FixedTensorDef(b_shape, dtype=dtype)):
        with flow.device_prior_placement(device_type, "0:0"):
            return func(a, b)

    a = np_array(dtype, a_shape)
    b = np_array(dtype, b_shape)

    y = ModJob1(a).get().ndarray()
    test_case.assertTrue(np.array_equal(y, func(a, a)))

    y = ModJob2(a, b).get().ndarray()
    test_case.assertTrue(np.array_equal(y, func(a, b)))

    flow.clear_default_session()

def test_broadcast_logical(test_case):
    arg_dict = OrderedDict()
    arg_dict["test_case"] = [test_case]
    arg_dict["func"] = [func_equal, func_not_equal, func_greater_than, func_greater_equal, func_less_than, func_less_than]
    arg_dict["a_shape"] = [(64, 64), (64, 64, 64)]
    arg_dict["b_shape"] = [(1, 64), (64, 1), (64, 1, 64), (1, 64, 1)]
    arg_dict["data_type"] = [flow.int8, flow.int32, flow.int64, flow.float, flow.double]
    arg_dict["device_type"] = ["cpu", "gpu"]

    for arg in GenArgList(arg_dict):
        if arg[5] == "cpu" and arg[4] == "float16": continue
        if len(arg[2]) < len(arg[3]): continue
        GenerateTest(*arg)

def test_xy_mod_x1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.int8)

def test_xy_mod_1y(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64))

def test_xyz_mod_x1z(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64))

def test_xyz_mod_1y1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1))

