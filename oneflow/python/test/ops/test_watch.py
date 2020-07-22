import numpy as np
import oneflow as flow
import oneflow.typing as oft


def test_simple(test_case):
    flow.config.gpu_device_num(1)
    data = np.ones((10,), dtype=np.float32)

    def EqOnes(x):
        test_case.assertTrue(np.allclose(data, x.numpy()))

    @flow.global_function()
    def ReluJob(x: oft.Numpy.Placeholder((10,))):
        flow.watch(x, EqOnes)

    ReluJob(data)


def test_two_device(test_case):
    flow.config.gpu_device_num(2)
    data = np.ones((10,), dtype=np.float32)

    def EqOnes(x):
        test_case.assertTrue(np.allclose(data, x.numpy()))

    @flow.global_function()
    def ReluJob(x: oft.Numpy.Placeholder((10,))):
        y = flow.math.relu(x)
        flow.watch(y, EqOnes)

    ReluJob(data)
