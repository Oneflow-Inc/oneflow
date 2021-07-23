import unittest
from typing import Tuple

import numpy as np

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


@flow.unittest.skip_unless_1n4d()
class TestFunctionInputOutput(flow.unittest.TestCase):
    def test_FixedTensorDef(test_case):
        @flow.global_function()
        def Foo(x: oft.Numpy.Placeholder((2, 5))):
            return x

        data = np.ones((2, 5), dtype=np.float32)
        of_ret = Foo(data).get()
        test_case.assertEqual(of_ret.numpy().max(), 1)
        test_case.assertEqual(of_ret.numpy().min(), 1)
        test_case.assertTrue(np.allclose(of_ret.numpy(), data))

    def test_FixedTensorDef_2_device(test_case):
        flow.config.gpu_device_num(2)

        @flow.global_function()
        def Foo(x: oft.Numpy.Placeholder((2, 5))):
            return x

        data = np.ones((2, 5), dtype=np.float32)
        of_ret = Foo(data).get()
        test_case.assertEqual(of_ret.numpy().max(), 1)
        test_case.assertEqual(of_ret.numpy().min(), 1)
        test_case.assertTrue(np.allclose(of_ret.numpy(), data))

    def test_MirroredTensorDef(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.ListNumpy.Placeholder((2, 5))):
            return x

        data = np.ones((1, 5), dtype=np.float32)
        ndarray_list = Foo([data]).get().numpy_list()
        test_case.assertEqual(len(ndarray_list), 1)
        test_case.assertTrue(np.allclose(ndarray_list[0], data))


if __name__ == "__main__":
    unittest.main()
