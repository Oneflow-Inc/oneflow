import oneflow as flow
import numpy as np
import os

def MakeTestConstant(test_case):
    def TestConstant():
        x = flow.constant(0, shape=(10,), dtype=flow.float)
        test_case.assertTrue(np.array_equal(x.numpy(), np.zeros((10,), dtype=np.float32)))
    return TestConstant

def WithExplicitPlacementScope(device, test_func):
    if os.getenv('ENABLE_USER_OP') != 'True': return
    with flow.device(device): test_func()

def WithDefaultPlacementScope(test_func):
    if os.getenv('ENABLE_USER_OP') != 'True': return
    test_func()


def test_cpu_constant(test_case):
    WithExplicitPlacementScope("cpu:0", MakeTestConstant(test_case))


def test_gpu_constant(test_case):
    WithExplicitPlacementScope("gpu:0", MakeTestConstant(test_case))


def test_default_placement_constant(test_case):
    WithDefaultPlacementScope(MakeTestConstant(test_case))
