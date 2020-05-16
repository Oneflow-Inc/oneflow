import oneflow as flow
import numpy as np

def TestConstant(test_case, device_tag):
    with flow.device("%s:0"%device_tag):
        x = flow.constant(0, shape=(10,), dtype=flow.float)
        test_case.assertTrue(np.array_equal(x.numpy(), np.zeros((10,), dtype=np.float32)))

def test_cpu_constant(test_case):
    TestConstant(test_case, 'cpu')


def test_gpu_constant(test_case):
    TestConstant(test_case, 'gpu')
