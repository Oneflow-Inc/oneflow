import oneflow as flow
import numpy as np

def test_simple(test_case):
    flow.config.gpu_device_num(1)
    data = np.ones((10,), dtype=np.float32)
    def EqOnes(x):
        test_case.assertTrue(np.allclose(data, x.ndarray()))
    @flow.function()
    def ReluJob(x = flow.FixedTensorDef((10,))): flow.watch(x, EqOnes)
    ReluJob(data)

def test_two_device(test_case):
    flow.config.gpu_device_num(2)
    data = np.ones((10,), dtype=np.float32)
    def EqOnes(x):
        test_case.assertTrue(np.allclose(data, x.ndarray()))
    @flow.function()
    def ReluJob(x = flow.FixedTensorDef((10,))):
        flow.watch(flow.math.relu(x), EqOnes)
    ReluJob(data)
