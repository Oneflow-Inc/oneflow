import oneflow as flow
import numpy as np

def test_asinh(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AsinhJob(a=flow.FixedTensorDef((8,))):
        return flow.math.asinh(a)

    x = np.array([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
    y = AsinhJob(x).get().ndarray()
    # output: [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
    test_case.assertTrue(np.allclose(y, np.arcsinh(x), equal_nan=True))
    print("asinh y = ", y)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = AsinhJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arcsinh(x), equal_nan=True))

