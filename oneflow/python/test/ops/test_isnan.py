import oneflow as flow
import numpy as np

def test_isnan(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def IsnanJob(a=flow.FixedTensorDef((9,))):
        return flow.math.isnan(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf"), -float("nan"), float("nan")], dtype=np.float32)
    y = IsnanJob(x).get().ndarray()
 
    print("isnan y = ", y) 
    print("np.isnan y =", np.isnan(x)) 
    test_case.assertTrue(np.allclose(y, np.isnan(x)))

    x = np.random.uniform(size=(9,)).astype(np.float32)
    y = IsnanJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.isnan(x)))