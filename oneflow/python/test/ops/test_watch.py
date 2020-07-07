import numpy as np
import oneflow as flow

def test_simple(test_case):
    _test_simple(False, test_case)

def test_simple_eager(test_case):
    _test_simple(True, test_case)

def _test_simple(enable_eager_execution, test_case):
    flow.enable_eager_execution(enable_eager_execution)
    flow.config.gpu_device_num(1)
    data = np.ones((10,), dtype=np.float32)

    def EqOnes(x):
        test_case.assertTrue(np.allclose(data, x.ndarray()))

    @flow.global_function()
    def ReluJob(x=flow.FixedTensorDef((10,))):
        flow.watch(x, EqOnes)

    ReluJob(data)

def test_two_device(test_case):
    _test_two_device(False, test_case)

def test_two_device_eager(test_case):
    _test_two_device(True, test_case)

def _test_two_device(enable_eager_execution, test_case):
    flow.enable_eager_execution(enable_eager_execution)
    flow.config.gpu_device_num(2)
    data = np.ones((10,), dtype=np.float32)

    def EqOnes(x):
        test_case.assertTrue(np.allclose(data, x.ndarray()))

    @flow.global_function()
    def ReluJob(x=flow.FixedTensorDef((10,))):
        flow.watch(flow.math.relu(x), EqOnes)

    ReluJob(data)
