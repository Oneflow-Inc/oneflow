import numpy as np
import oneflow as flow
import oneflow.typing as oft


def test_bernoulli(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def BernoulliJob(a: oft.Numpy.Placeholder((10, 10))):
        return flow.random.bernoulli(a)

    x = np.ones((10, 10), dtype=np.float32)
    y = BernoulliJob(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, x))

    x = np.zeros((10, 10), dtype=np.float32)
    y = BernoulliJob(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, x))


def test_bernoulli_mirrored(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def BernoulliJob(a: oft.ListNumpy.Placeholder((10, 10))):
        return flow.random.bernoulli(a)

    x = np.ones((10, 10), dtype=np.float32)
    y = BernoulliJob([x]).get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, x))

    x = np.zeros((10, 10), dtype=np.float32)
    y = BernoulliJob([x]).get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, x))
