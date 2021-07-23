import unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
import numpy as np

@flow.unittest.skip_unless_1n2d()
class TestDemoMatmul(flow.unittest.TestCase):

    def test_watch(test_case):
        flow.config.gpu_device_num(2)
        flow.config.enable_debug_mode(True)
        expected = np.array([[30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 30]]).astype(np.float32)

        def Watch(x: tp.Numpy):
            test_case.assertTrue(np.allclose(x, expected))

        @flow.global_function()
        def Matmul(x: tp.Numpy.Placeholder((4, 4), dtype=flow.float32), y: tp.Numpy.Placeholder((4, 4), dtype=flow.float32)) -> tp.Numpy:
            s = flow.matmul(x, y)
            flow.watch(s, Watch)
            z = flow.matmul(s, x)
            return z
        x = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]).astype(np.float32)
        y = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]).astype(np.float32)
        Matmul(x, y)
if __name__ == '__main__':
    unittest.main()