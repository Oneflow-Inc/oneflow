import oneflow as flow
import numpy as np


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


def GenerateTest(test_case, func, a_shape, b_shape, dtype=flow.int32):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)

    @flow.function(func_config)
    def ModJob1(a=flow.FixedTensorDef(a_shape, dtype=dtype)):
        return func(a, a)

    @flow.function(func_config)
    def ModJob2(a=flow.FixedTensorDef(a_shape, dtype=dtype),
               b=flow.FixedTensorDef(b_shape, dtype=dtype)):
        return func(a, b)

    a = np_array(dtype, a_shape)
    b = np_array(dtype, b_shape)

    y = ModJob1(a).get().ndarray()
    test_case.assertTrue(np.array_equal(y, func(a, a)))

    y = ModJob2(a, b).get().ndarray()
    test_case.assertTrue(np.array_equal(y, func(a, b)))

    flow.clear_default_session()


def test_xy_mod_x1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.int8)

def test_xy_mod_1y(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64))

def test_xyz_mod_x1z(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64))

def test_xyz_mod_1y1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1))

def test_equal_int8_shape0(test_case):
    GenerateTest(test_case, func_equal, (64,), (1, ), flow.int8)
def test_equal_int8_shape1(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (64, 1), flow.int8)
def test_equal_int8_shape2(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (1, 64), flow.int8)
def test_equal_int8_shape3(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64), flow.int8)
def test_equal_int8_shape4(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (1, 64, 1), flow.int8)
def test_equal_int32_shape0(test_case):
    GenerateTest(test_case, func_equal, (64,), (1, ), flow.int32)
def test_equal_int32_shape1(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (64, 1), flow.int32)
def test_equal_int32_shape2(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (1, 64), flow.int32)
def test_equal_int32_shape3(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64), flow.int32)
def test_equal_int32_shape4(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (1, 64, 1), flow.int32)
def test_equal_int64_shape0(test_case):
    GenerateTest(test_case, func_equal, (64,), (1, ), flow.int64)
def test_equal_int64_shape1(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (64, 1), flow.int64)
def test_equal_int64_shape2(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (1, 64), flow.int64)
def test_equal_int64_shape3(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64), flow.int64)
def test_equal_int64_shape4(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (1, 64, 1), flow.int64)
def test_equal_float_shape0(test_case):
    GenerateTest(test_case, func_equal, (64,), (1, ), flow.float)
def test_equal_float_shape1(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (64, 1), flow.float)
def test_equal_float_shape2(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (1, 64), flow.float)
def test_equal_float_shape3(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64), flow.float)
def test_equal_float_shape4(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (1, 64, 1), flow.float)
def test_equal_double_shape0(test_case):
    GenerateTest(test_case, func_equal, (64,), (1, ), flow.double)
def test_equal_double_shape1(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (64, 1), flow.double)
def test_equal_double_shape2(test_case):
    GenerateTest(test_case, func_equal, (64, 64), (1, 64), flow.double)
def test_equal_double_shape3(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (64, 1, 64), flow.double)
def test_equal_double_shape4(test_case):
    GenerateTest(test_case, func_equal, (64, 64, 64), (1, 64, 1), flow.double)
def test_not_equal_int8_shape0(test_case):
    GenerateTest(test_case, func_not_equal, (64,), (1, ), flow.int8)
def test_not_equal_int8_shape1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (64, 1), flow.int8)
def test_not_equal_int8_shape2(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (1, 64), flow.int8)
def test_not_equal_int8_shape3(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (64, 1, 64), flow.int8)
def test_not_equal_int8_shape4(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1), flow.int8)
def test_not_equal_int32_shape0(test_case):
    GenerateTest(test_case, func_not_equal, (64,), (1, ), flow.int32)
def test_not_equal_int32_shape1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (64, 1), flow.int32)
def test_not_equal_int32_shape2(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (1, 64), flow.int32)
def test_not_equal_int32_shape3(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (64, 1, 64), flow.int32)
def test_not_equal_int32_shape4(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1), flow.int32)
def test_not_equal_int64_shape0(test_case):
    GenerateTest(test_case, func_not_equal, (64,), (1, ), flow.int64)
def test_not_equal_int64_shape1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (64, 1), flow.int64)
def test_not_equal_int64_shape2(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (1, 64), flow.int64)
def test_not_equal_int64_shape3(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (64, 1, 64), flow.int64)
def test_not_equal_int64_shape4(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1), flow.int64)
def test_not_equal_float_shape0(test_case):
    GenerateTest(test_case, func_not_equal, (64,), (1, ), flow.float)
def test_not_equal_float_shape1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (64, 1), flow.float)
def test_not_equal_float_shape2(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (1, 64), flow.float)
def test_not_equal_float_shape3(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (64, 1, 64), flow.float)
def test_not_equal_float_shape4(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1), flow.float)
def test_not_equal_double_shape0(test_case):
    GenerateTest(test_case, func_not_equal, (64,), (1, ), flow.double)
def test_not_equal_double_shape1(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (64, 1), flow.double)
def test_not_equal_double_shape2(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64), (1, 64), flow.double)
def test_not_equal_double_shape3(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (64, 1, 64), flow.double)
def test_not_equal_double_shape4(test_case):
    GenerateTest(test_case, func_not_equal, (64, 64, 64), (1, 64, 1), flow.double)
def test_greater_than_int8_shape0(test_case):
    GenerateTest(test_case, func_greater_than, (64,), (1, ), flow.int8)
def test_greater_than_int8_shape1(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (64, 1), flow.int8)
def test_greater_than_int8_shape2(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64), flow.int8)
def test_greater_than_int8_shape3(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (64, 1, 64), flow.int8)
def test_greater_than_int8_shape4(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (1, 64, 1), flow.int8)
def test_greater_than_int32_shape0(test_case):
    GenerateTest(test_case, func_greater_than, (64,), (1, ), flow.int32)
def test_greater_than_int32_shape1(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (64, 1), flow.int32)
def test_greater_than_int32_shape2(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64), flow.int32)
def test_greater_than_int32_shape3(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (64, 1, 64), flow.int32)
def test_greater_than_int32_shape4(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (1, 64, 1), flow.int32)
def test_greater_than_int64_shape0(test_case):
    GenerateTest(test_case, func_greater_than, (64,), (1, ), flow.int64)
def test_greater_than_int64_shape1(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (64, 1), flow.int64)
def test_greater_than_int64_shape2(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64), flow.int64)
def test_greater_than_int64_shape3(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (64, 1, 64), flow.int64)
def test_greater_than_int64_shape4(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (1, 64, 1), flow.int64)
def test_greater_than_float_shape0(test_case):
    GenerateTest(test_case, func_greater_than, (64,), (1, ), flow.float)
def test_greater_than_float_shape1(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (64, 1), flow.float)
def test_greater_than_float_shape2(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64), flow.float)
def test_greater_than_float_shape3(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (64, 1, 64), flow.float)
def test_greater_than_float_shape4(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (1, 64, 1), flow.float)
def test_greater_than_double_shape0(test_case):
    GenerateTest(test_case, func_greater_than, (64,), (1, ), flow.double)
def test_greater_than_double_shape1(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (64, 1), flow.double)
def test_greater_than_double_shape2(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64), (1, 64), flow.double)
def test_greater_than_double_shape3(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (64, 1, 64), flow.double)
def test_greater_than_double_shape4(test_case):
    GenerateTest(test_case, func_greater_than, (64, 64, 64), (1, 64, 1), flow.double)
def test_greater_equal_int8_shape0(test_case):
    GenerateTest(test_case, func_greater_equal, (64,), (1, ), flow.int8)
def test_greater_equal_int8_shape1(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (64, 1), flow.int8)
def test_greater_equal_int8_shape2(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (1, 64), flow.int8)
def test_greater_equal_int8_shape3(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (64, 1, 64), flow.int8)
def test_greater_equal_int8_shape4(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (1, 64, 1), flow.int8)
def test_greater_equal_int32_shape0(test_case):
    GenerateTest(test_case, func_greater_equal, (64,), (1, ), flow.int32)
def test_greater_equal_int32_shape1(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (64, 1), flow.int32)
def test_greater_equal_int32_shape2(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (1, 64), flow.int32)
def test_greater_equal_int32_shape3(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (64, 1, 64), flow.int32)
def test_greater_equal_int32_shape4(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (1, 64, 1), flow.int32)
def test_greater_equal_int64_shape0(test_case):
    GenerateTest(test_case, func_greater_equal, (64,), (1, ), flow.int64)
def test_greater_equal_int64_shape1(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (64, 1), flow.int64)
def test_greater_equal_int64_shape2(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (1, 64), flow.int64)
def test_greater_equal_int64_shape3(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (64, 1, 64), flow.int64)
def test_greater_equal_int64_shape4(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (1, 64, 1), flow.int64)
def test_greater_equal_float_shape0(test_case):
    GenerateTest(test_case, func_greater_equal, (64,), (1, ), flow.float)
def test_greater_equal_float_shape1(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (64, 1), flow.float)
def test_greater_equal_float_shape2(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (1, 64), flow.float)
def test_greater_equal_float_shape3(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (64, 1, 64), flow.float)
def test_greater_equal_float_shape4(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (1, 64, 1), flow.float)
def test_greater_equal_double_shape0(test_case):
    GenerateTest(test_case, func_greater_equal, (64,), (1, ), flow.double)
def test_greater_equal_double_shape1(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (64, 1), flow.double)
def test_greater_equal_double_shape2(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64), (1, 64), flow.double)
def test_greater_equal_double_shape3(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (64, 1, 64), flow.double)
def test_greater_equal_double_shape4(test_case):
    GenerateTest(test_case, func_greater_equal, (64, 64, 64), (1, 64, 1), flow.double)
def test_less_than_int8_shape0(test_case):
    GenerateTest(test_case, func_less_than, (64,), (1, ), flow.int8)
def test_less_than_int8_shape1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.int8)
def test_less_than_int8_shape2(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (1, 64), flow.int8)
def test_less_than_int8_shape3(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (64, 1, 64), flow.int8)
def test_less_than_int8_shape4(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (1, 64, 1), flow.int8)
def test_less_than_int32_shape0(test_case):
    GenerateTest(test_case, func_less_than, (64,), (1, ), flow.int32)
def test_less_than_int32_shape1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.int32)
def test_less_than_int32_shape2(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (1, 64), flow.int32)
def test_less_than_int32_shape3(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (64, 1, 64), flow.int32)
def test_less_than_int32_shape4(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (1, 64, 1), flow.int32)
def test_less_than_int64_shape0(test_case):
    GenerateTest(test_case, func_less_than, (64,), (1, ), flow.int64)
def test_less_than_int64_shape1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.int64)
def test_less_than_int64_shape2(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (1, 64), flow.int64)
def test_less_than_int64_shape3(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (64, 1, 64), flow.int64)
def test_less_than_int64_shape4(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (1, 64, 1), flow.int64)
def test_less_than_float_shape0(test_case):
    GenerateTest(test_case, func_less_than, (64,), (1, ), flow.float)
def test_less_than_float_shape1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.float)
def test_less_than_float_shape2(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (1, 64), flow.float)
def test_less_than_float_shape3(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (64, 1, 64), flow.float)
def test_less_than_float_shape4(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (1, 64, 1), flow.float)
def test_less_than_double_shape0(test_case):
    GenerateTest(test_case, func_less_than, (64,), (1, ), flow.double)
def test_less_than_double_shape1(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (64, 1), flow.double)
def test_less_than_double_shape2(test_case):
    GenerateTest(test_case, func_less_than, (64, 64), (1, 64), flow.double)
def test_less_than_double_shape3(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (64, 1, 64), flow.double)
def test_less_than_double_shape4(test_case):
    GenerateTest(test_case, func_less_than, (64, 64, 64), (1, 64, 1), flow.double)
def test_less_equal_int8_shape0(test_case):
    GenerateTest(test_case, func_less_equal, (64,), (1, ), flow.int8)
def test_less_equal_int8_shape1(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (64, 1), flow.int8)
def test_less_equal_int8_shape2(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (1, 64), flow.int8)
def test_less_equal_int8_shape3(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (64, 1, 64), flow.int8)
def test_less_equal_int8_shape4(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (1, 64, 1), flow.int8)
def test_less_equal_int32_shape0(test_case):
    GenerateTest(test_case, func_less_equal, (64,), (1, ), flow.int32)
def test_less_equal_int32_shape1(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (64, 1), flow.int32)
def test_less_equal_int32_shape2(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (1, 64), flow.int32)
def test_less_equal_int32_shape3(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (64, 1, 64), flow.int32)
def test_less_equal_int32_shape4(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (1, 64, 1), flow.int32)
def test_less_equal_int64_shape0(test_case):
    GenerateTest(test_case, func_less_equal, (64,), (1, ), flow.int64)
def test_less_equal_int64_shape1(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (64, 1), flow.int64)
def test_less_equal_int64_shape2(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (1, 64), flow.int64)
def test_less_equal_int64_shape3(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (64, 1, 64), flow.int64)
def test_less_equal_int64_shape4(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (1, 64, 1), flow.int64)
def test_less_equal_float_shape0(test_case):
    GenerateTest(test_case, func_less_equal, (64,), (1, ), flow.float)
def test_less_equal_float_shape1(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (64, 1), flow.float)
def test_less_equal_float_shape2(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (1, 64), flow.float)
def test_less_equal_float_shape3(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (64, 1, 64), flow.float)
def test_less_equal_float_shape4(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (1, 64, 1), flow.float)
def test_less_equal_double_shape0(test_case):
    GenerateTest(test_case, func_less_equal, (64,), (1, ), flow.double)
def test_less_equal_double_shape1(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (64, 1), flow.double)
def test_less_equal_double_shape2(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64), (1, 64), flow.double)
def test_less_equal_double_shape3(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (64, 1, 64), flow.double)
def test_less_equal_double_shape4(test_case):
    GenerateTest(test_case, func_less_equal, (64, 64, 64), (1, 64, 1), flow.double)



