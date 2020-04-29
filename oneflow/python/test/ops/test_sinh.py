import oneflow as flow
import numpy as np

def test_sinh(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def SinhJob(a=flow.FixedTensorDef((8,))):
        return flow.math.sinh(a)

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    y = SinhJob(x).get().ndarray()
    # output: [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
    test_case.assertTrue(np.allclose(y, np.sinh(x), equal_nan=True))

    x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
    y = SinhJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.sinh(x), equal_nan=True))

