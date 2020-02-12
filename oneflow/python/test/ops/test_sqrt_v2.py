import oneflow as flow
import numpy as np

def test_sqrt_v2(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def SqrtJob(a=flow.FixedTensorDef((8,))):
        return flow.math.sqrt_v2(a)

    x = np.random.uniform(low=0.0, high=100.0, size=(8,)).astype(np.float32)
    y = SqrtJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.sqrt(x), equal_nan=True))

