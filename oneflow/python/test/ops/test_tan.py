import oneflow as flow
import numpy as np

def test_tan(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def TanJob(a=flow.FixedTensorDef((8,))):
        return flow.math.tan(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
    y = TanJob(x).get().ndarray()
    # output: [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
    test_case.assertTrue(np.allclose(y, np.tan(x), equal_nan=True))

    x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
    y = TanJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.tan(x), equal_nan=True))

