import oneflow as flow
import numpy as np

def test_sign(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def SignJob(a=flow.FixedTensorDef((8,))):
        return flow.math.sign(a)

    x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
    y = SignJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.sign(x), equal_nan=True))

