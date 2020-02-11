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


