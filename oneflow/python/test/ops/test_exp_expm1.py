import oneflow as flow
import numpy as np

def test_exp(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ExpJob(a=flow.FixedTensorDef((8,))):
        return flow.math.exp(a)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = ExpJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.exp(x), equal_nan=True))

def test_expm1(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def Expm1Job(a=flow.FixedTensorDef((8,))):
        return flow.math.expm1(a)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = Expm1Job(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.expm1(x), equal_nan=True))

