import oneflow as flow
import numpy as np

def test_square_v2(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def SquareJob(a=flow.FixedTensorDef((8,))):
        return flow.math.square_v2(a)

    x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
    y = SquareJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, x * x, equal_nan=True))

