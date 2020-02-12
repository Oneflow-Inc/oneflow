import oneflow as flow
import numpy as np

def test_rint(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def RintJob(a=flow.FixedTensorDef((8,))):
        return flow.math.rint(a)

    x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
    y = RintJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.rint(x), equal_nan=True))

def test_rint_special_value(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def RintJob(a=flow.FixedTensorDef((9,))):
        return flow.math.rint(a)

    x = np.array([0.5000001, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.5, 3.5], dtype=np.float32)
    out = np.array([1.0, -2., -2., -0., 0., 2., 2., 2., 4.], dtype=np.float32)
    y = RintJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, out, equal_nan=True))

def test_round(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def RoundJob(a=flow.FixedTensorDef((8,))):
        return flow.math.round(a)

    x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
    y = RoundJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.round(x), equal_nan=True))

def test_round_special_value(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def RoundJob(a=flow.FixedTensorDef((5,))):
        return flow.math.round(a)

    x = np.array([0.9, 2.5, 2.3, 1.5, -4.5], dtype=np.float32)
    out = np.array([1.0, 2.0, 2.0, 2.0, -4.0], dtype=np.float32)
    y = RoundJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, out, equal_nan=True))


