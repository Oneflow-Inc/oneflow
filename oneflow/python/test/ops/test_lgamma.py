import oneflow as flow
import numpy as np
from scipy.special import gammaln

def test_lgamma(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def LgammaJob(a=flow.FixedTensorDef((6,))):
        return flow.math.lgamma(a)

    x = np.array([0, 0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
    y = LgammaJob(x).get().ndarray()
    # output: [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
    # print("lgamma y = ", y)
    test_case.assertTrue(np.allclose(y, gammaln(x), equal_nan=True))

