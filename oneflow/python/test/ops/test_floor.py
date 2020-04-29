import oneflow as flow
import numpy as np

def test_floor(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def FloorJob(a=flow.FixedTensorDef((8,))):
        return flow.math.floor(a)

    x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
    y = FloorJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.floor(x), equal_nan=True))

