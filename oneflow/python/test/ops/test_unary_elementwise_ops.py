import oneflow as flow
import numpy as np

def test_abs(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AbsJob(a=flow.FixedTensorDef((5, 2))):
        return flow.math.abs(a)

    x = np.random.rand(5, 2).astype(np.float32)
    y = AbsJob(x).get().ndarray()
    test_case.assertTrue(np.array_equal(y, np.absolute(x)))

def test_1n2c_mirror_dynamic_abs(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def AbsJob(a = flow.MirroredTensorDef((5, 2))):
        return flow.math.abs(a)

    x1 = np.random.rand(3, 1).astype(np.float32)
    x2 = np.random.rand(4, 2).astype(np.float32)
    y1, y2 = AbsJob([x1, x2]).get().ndarray_list()
    test_case.assertTrue(np.array_equal(y1, np.absolute(x1)))
    test_case.assertTrue(np.array_equal(y2, np.absolute(x2)))

def test_acos(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AcosJob(a=flow.FixedTensorDef((5, 2))):
        return flow.math.acos(a)

    x = np.random.rand(5, 2).astype(np.float32)
    y = AcosJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arccos(x)))

def test_1n2c_mirror_dynamic_acos(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def AcosJob(a=flow.MirroredTensorDef((5, 2))):
        return flow.math.acos(a)

    x1 = np.random.rand(3, 1).astype(np.float32)
    x2 = np.random.rand(4, 2).astype(np.float32)
    y1, y2 = AcosJob([x1, x2]).get().ndarray_list()
    test_case.assertTrue(np.allclose(y1, np.arccos(x1)))
    test_case.assertTrue(np.allclose(y2, np.arccos(x2)))

