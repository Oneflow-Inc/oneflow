import numpy as np
import oneflow as flow
import oneflow.typing as oft


def test_scalar_div_2(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def Div2Job(a: oft.Numpy.Placeholder((10, 10))):
        return a / 2

    x = np.random.rand(10, 10).astype(np.float32) + 1
    y = Div2Job(x).get().numpy()
    test_case.assertTrue(np.allclose(y, x / 2))


def test_scalar_div_by_2(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def DivBy2Job(a: oft.Numpy.Placeholder((10, 10))):
        return 2 / a

    x = np.random.rand(10, 10).astype(np.float32) + 1
    y = DivBy2Job(x).get().numpy()
    test_case.assertTrue(np.allclose(y, 2 / x))


def test_scalar_div_2_mirrored(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def Div2Job(a: oft.ListNumpy.Placeholder((10, 10))):
        return a / 2

    x = np.random.rand(10, 10).astype(np.float32) + 1
    y = Div2Job([x]).get().numpy_list()[0]
    test_case.assertTrue(np.allclose(y, x / 2))


def test_scalar_div_by_2_mirrored(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def DivBy2Job(a: oft.ListNumpy.Placeholder((10, 10))):
        return 2 / a

    x = np.random.rand(10, 10).astype(np.float32) + 1
    y = DivBy2Job([x]).get().numpy_list()[0]
    test_case.assertTrue(np.allclose(y, 2 / x))
