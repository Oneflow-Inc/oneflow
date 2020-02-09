import oneflow as flow
import numpy as np

def test_acos(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AcosJob(a=flow.FixedTensorDef((5, 2))):
        return flow.math.acos(a)

    x = np.random.rand(5, 2).astype(np.float32)
    y = AcosJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arccos(x)))

def test_mirror_acos(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def AcosJob(a=flow.FixedTensorDef((5, 2))):
        return flow.math.acos(a)

    x = np.random.rand(5, 2).astype(np.float32)
    y = AcosJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arccos(x)))

