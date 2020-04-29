import oneflow as flow
import numpy as np

def test_reciprocal(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ReciprocalJob(a=flow.FixedTensorDef((8,))):
        return flow.math.reciprocal(a)

    x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
    y = ReciprocalJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, 1.0 / x, equal_nan=True))

def test_reciprocal_no_nan(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ReciprocalNoNanJob(a=flow.FixedTensorDef((4,))):
        return flow.math.reciprocal_no_nan(a)

    x = np.array([2.0, 0.5, 0, 1], dtype=np.float32)
    out = np.array([0.5, 2, 0.0, 1.0], dtype=np.float32)
    y = ReciprocalNoNanJob(x).get().ndarray()
    # print("reciprocal_no_nan: y = ", y)
    test_case.assertTrue(np.allclose(y, out, equal_nan=True))


