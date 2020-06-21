import numpy as np
import oneflow as flow

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_naive(test_case):
    @flow.global_function(func_config)
    def ModJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((5, 2))):
        return a % b

    x = np.random.rand(5, 2).astype(np.float32)
    y = np.random.rand(5, 2).astype(np.float32)
    z = None
    z = ModJob(x, y).get().ndarray()
    test_case.assertTrue(np.allclose(z, x % y))


def test_broadcast(test_case):
    @flow.global_function(func_config)
    def ModJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((1, 2))):
        return a % b

    x = np.random.rand(5, 2).astype(np.float32)
    y = np.random.rand(1, 2).astype(np.float32)
    z = None
    z = ModJob(x, y).get().ndarray()
    test_case.assertTrue(np.allclose(z, x % y))


def test_xy_mod_x1(test_case):
    GenerateTest(test_case, (64, 64), (64, 1))


def test_xy_mod_1y(test_case):
    GenerateTest(test_case, (64, 64), (1, 64))


def test_xyz_mod_x1z(test_case):
    GenerateTest(test_case, (64, 64, 64), (64, 1, 64))


def test_xyz_mod_1y1(test_case):
    GenerateTest(test_case, (64, 64, 64), (1, 64, 1))


def GenerateTest(test_case, a_shape, b_shape):
    @flow.global_function(func_config)
    def ModJob(a=flow.FixedTensorDef(a_shape), b=flow.FixedTensorDef(b_shape)):
        return a % b

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)
    y = ModJob(a, b).get().ndarray()
    test_case.assertTrue(np.allclose(y, a % b))
