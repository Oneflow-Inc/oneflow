import oneflow as flow
import numpy as np

def test_isinf(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def IsinfJob(a=flow.FixedTensorDef((9,))):
        return flow.math.isinf(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf"), float("nan")], dtype=np.float32)
    y = IsinfJob(x).get().ndarray()
    
    print("isinf y = ", y)
    print("np.isinf y =", np.isinf(x))
    test_case.assertTrue(np.allclose(y, np.isinf(x)))

    x = np.random.uniform(size=(9,)).astype(np.float32)
    y = IsinfJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.isinf(x)))