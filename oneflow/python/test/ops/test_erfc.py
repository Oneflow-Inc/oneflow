import oneflow as flow
import numpy as np
from scipy.special import erfc

def test_erfc(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ErfcJob(a=flow.FixedTensorDef((8,))):
        return flow.math.erfc(a)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = ErfcJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, erfc(x), equal_nan=True))

