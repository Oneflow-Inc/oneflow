import oneflow as flow
import numpy as np
from scipy.special import erf

def test_erf(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ErfJob(a=flow.FixedTensorDef((8,))):
        return flow.math.erf(a)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = ErfJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, erf(x), equal_nan=True))

