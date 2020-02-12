import oneflow as flow
import numpy as np

def test_sigmoid_v2(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def SigmoidJob(a=flow.FixedTensorDef((8,))):
        return flow.math.sigmoid_v2(a)

    x = np.random.uniform(low=-2.0, high=2.0, size=(8,)).astype(np.float32)
    y = SigmoidJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, 1.0 / (1.0 + np.exp(-x)), equal_nan=True))

