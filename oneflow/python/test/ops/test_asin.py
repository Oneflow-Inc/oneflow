import oneflow as flow
import numpy as np

def test_asin(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AsinJob(a=flow.FixedTensorDef((2,))):
        return flow.math.asin(a)

    x = np.array([0.8659266, 0.7068252], dtype=np.float32)
    y = AsinJob(x).get().ndarray()
    # output: [1.047, 0.785] ~= [(PI/3), (PI/4)]
    test_case.assertTrue(np.allclose(y, np.arcsin(x), equal_nan=True))

    x = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)
    y = AsinJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arcsin(x), equal_nan=True))

