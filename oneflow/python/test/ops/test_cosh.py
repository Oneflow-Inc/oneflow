import oneflow as flow
import numpy as np

def test_cosh(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def CoshJob(a=flow.FixedTensorDef((8,))):
        return flow.math.cosh(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    y = CoshJob(x).get().ndarray()
    # output: [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
    test_case.assertTrue(np.allclose(y, np.cosh(x), equal_nan=True))
    # print("cosh y = ", y)

    x = np.random.uniform(size=(8,)).astype(np.float32)
    y = CoshJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.cosh(x), equal_nan=True))

