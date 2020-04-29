import oneflow as flow
import numpy as np

def test_sin(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def SinJob(a=flow.FixedTensorDef((8,))):
        return flow.math.sin(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")], dtype=np.float32)
    y = SinJob(x).get().ndarray()
    # output: [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
    test_case.assertTrue(np.allclose(y, np.sin(x), equal_nan=True))

    x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
    y = SinJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.sin(x), equal_nan=True))

