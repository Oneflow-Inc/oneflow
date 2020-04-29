import oneflow as flow
import numpy as np

def test_tanh_v2(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def TanhJob(a=flow.FixedTensorDef((8,))):
        return flow.math.tanh_v2(a)

    x = np.array([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")], dtype=np.float32)
    y = TanhJob(x).get().ndarray()
    # output: [-1. -0.99990916 -0.46211717 0.7615942 0.8336547 0.9640276 0.9950547 1.]
    test_case.assertTrue(np.allclose(y, np.tanh(x), equal_nan=True))

    x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
    y = TanhJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.tanh(x), equal_nan=True))

