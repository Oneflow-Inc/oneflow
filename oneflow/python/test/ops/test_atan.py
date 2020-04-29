import oneflow as flow
import numpy as np

def test_atan(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AtanJob(a=flow.FixedTensorDef((2,))):
        return flow.math.atan(a)

    x = np.array([1.731261, 0.99920404], dtype=np.float32)
    y = AtanJob(x).get().ndarray()
    # output: [1.047, 0.785] ~= [(PI/3), (PI/4)]
    test_case.assertTrue(np.allclose(y, np.arctan(x), equal_nan=True))
    # print("atan y = ", y)

    pi = 3.14159265357
    x = np.random.uniform(low=-pi/2, high=pi/2, size=(2,)).astype(np.float32)
    y = AtanJob(x).get().ndarray()
    test_case.assertTrue(np.allclose(y, np.arctan(x), equal_nan=True))

