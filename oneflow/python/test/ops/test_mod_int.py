import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.int32)
#func_config.default_data_type(flow.float32)

def test_naive(test_case):
    @flow.function(func_config)
    def ModJob(a=flow.FixedTensorDef((5, 2), dtype=flow.int32), b=flow.FixedTensorDef((5, 2), dtype=flow.int32)):
    #def ModJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((5, 2))):
        return a % b

    x = (np.random.rand(5, 2)*1000).astype(np.int32)+1
    y = (np.random.rand(5, 2)*1000).astype(np.int32)+1
    z = None
    z = ModJob(x, y).get().ndarray()
    test_case.assertTrue(np.array_equal(z, x % y))

def test_broadcast(test_case):
    @flow.function(func_config)
    def ModJob(a=flow.FixedTensorDef((5, 2), dtype=flow.int32), b=flow.FixedTensorDef((1, 2), dtype=flow.int32)):
        return a % b

    x = (np.random.rand(5, 2)*1000).astype(np.int32)+1
    y = (np.random.rand(1, 2)*1000).astype(np.int32)+1
    z = None
    z = ModJob(x, y).get().ndarray()
    test_case.assertTrue(np.array_equal(z, x % y))

def test_xy_mod_x1(test_case):
    GenerateTest(test_case, (64, 64), (64, 1))

def test_xy_mod_1y(test_case):
    GenerateTest(test_case, (64, 64), (1, 64))

def test_xyz_mod_x1z(test_case):
    GenerateTest(test_case, (64, 64, 64), (64, 1, 64))

def test_xyz_mod_1y1(test_case):
    GenerateTest(test_case, (64, 64, 64), (1, 64, 1))

def GenerateTest(test_case, a_shape, b_shape):
    @flow.function(func_config)
    def ModJob(a=flow.FixedTensorDef(a_shape, dtype=flow.int32), b=flow.FixedTensorDef(b_shape, dtype=flow.int32)):
        return a % b

    a = (np.random.rand(*a_shape)*1000).astype(np.int32)+1
    b = (np.random.rand(*b_shape)*1000).astype(np.int32)+1
    y = ModJob(a, b).get().ndarray()
    test_case.assertTrue(np.array_equal(y, a % b))
