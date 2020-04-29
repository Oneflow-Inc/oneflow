import oneflow as flow
import numpy as np

def test_cos(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def CosJob(a=flow.FixedTensorDef((8,))):
        return flow.math.cos(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
    y = CosJob(x).get().ndarray()
    # output: [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
    test_case.assertTrue(np.allclose(y, np.cos(x), equal_nan=True))
    # print("cos y = ", y)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = CosJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.cos(x), equal_nan=True))

