import oneflow as flow
import numpy as np

def test_acosh(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AcoshJob(a=flow.FixedTensorDef((7,))):
        return flow.math.acosh(a)

    # x = np.random.rand(7,).astype(np.float32)
    x = np.array([-2, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
    y = AcoshJob(x).get().ndarray()
    # input: [-2, -0.5, 1, 1.2, 200, 10000, float("inf")]
    # output: [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
    test_case.assertTrue(np.allclose(y, np.arccosh(x), equal_nan=True))

    x = np.random.uniform(low=1.0, high=100.0, size=(7,)).astype(np.float32)
    y = AcoshJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arccosh(x), equal_nan=True))

