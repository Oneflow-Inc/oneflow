import oneflow as flow
import numpy as np

def test_atanh(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AtanhJob(a=flow.FixedTensorDef((8,))):
        return flow.math.atanh(a)

    x = np.array([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")], dtype=np.float32)
    y = AtanhJob(x).get().ndarray()
    # output: [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
    test_case.assertTrue(np.allclose(y, np.arctanh(x), equal_nan=True))
    # print("atanh y = ", y)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = AtanhJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arctanh(x), equal_nan=True))

