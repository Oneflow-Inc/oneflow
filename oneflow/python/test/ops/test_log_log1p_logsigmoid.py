import oneflow as flow
import numpy as np
from scipy.special import gammaln

def test_log(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def LogJob(a=flow.FixedTensorDef((4,))):
        return flow.math.log(a)

    x = np.array([0, 0.5, 1, 5], dtype=np.float32)
    y = LogJob(x).get().ndarray()
    # output: [-inf, -0.6931472,  0. ,  1.609438]
    # print("log y = ", y)
    test_case.assertTrue(np.allclose(y, np.log(x), equal_nan=True))

def test_log1p(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def Log1pJob(a=flow.FixedTensorDef((4,))):
        return flow.math.log1p(a)

    x = np.array([0, 0.5, 1, 5], dtype=np.float32)
    y = Log1pJob(x).get().ndarray()
    # output: [0., 0.4054651, 0.6931472, 1.791759]
    # print("log1p y = ", y)
    test_case.assertTrue(np.allclose(y, np.log1p(x), equal_nan=True))

def test_log_sigmoid(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def LogSigmoidJob(a=flow.FixedTensorDef((8,))):
        return flow.math.log_sigmoid(a)

    x = np.random.uniform(low=-5.0, high=5.0, size=(8,)).astype(np.float32)
    y = LogSigmoidJob(x).get().ndarray()
    # print("log_sigmoid y = ", y)
    test_case.assertTrue(np.allclose(y, np.log(1 / (1 + np.exp(-x))), equal_nan=True))

