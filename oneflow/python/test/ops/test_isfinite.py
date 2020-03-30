import oneflow as flow
import numpy as np

def test_isfinite(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def IsfiniteJob(a=flow.FixedTensorDef((9,))):
        return flow.math.isfinite(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf"), float("nan")], dtype=np.float32)
    y = IsfiniteJob(x).get().ndarray()

    print("isfinite y = ", y) # int8 dtype output:  [0 1 1 1 1 1 1 0 0]
    print("np.isfinite y =", np.isfinite(x))  # output: [False True True True True True False False]
    test_case.assertTrue(np.allclose(y, np.isfinite(x)))

    x = np.random.uniform(size=(9,)).astype(np.float32)
    y = IsfiniteJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.isfinite(x)))