import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

@flow.unittest.skip_unless_1n2d()
class TestWatch(flow.unittest.TestCase):

    def test_simple(test_case):
        flow.config.gpu_device_num(1)
        data = np.ones((10,), dtype=np.float32)

        def EqOnes(x):
            test_case.assertTrue(np.allclose(data, x.numpy()))

        @flow.global_function()
        def ReluJob(x: oft.Numpy.Placeholder((10,))):
            flow.watch(x, EqOnes)
        ReluJob(data)

    @unittest.skipIf(flow.unittest.env.eager_execution_enabled(), "Doesn't work in eager mode")
    def test_two_device(test_case):
        flow.config.gpu_device_num(2)
        data = np.ones((10,), dtype=np.float32)

        def EqOnes(x):
            test_case.assertTrue(np.allclose(data, x.numpy()))
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def ReluJob(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.relu(x)
            flow.watch(y, EqOnes)
        ReluJob(data)
if __name__ == '__main__':
    unittest.main()