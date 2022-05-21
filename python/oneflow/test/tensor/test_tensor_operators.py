import unittest

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestTensorOperators(flow.unittest.TestCase):
    @autotest(n=5, check_graph=False)
    def test_tensor_inplace_operators_with_grad(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x_dptr = x.oneflow.data_ptr()
        x_id = id(x.oneflow)
        x -= 1
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        x /= 3
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        x += 5
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        x *= 7
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        return x

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_tensor_inplace_operators_without_grad(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x_dptr = x.oneflow.data_ptr()
        x_id = id(x.oneflow)
        x //= 2
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        return x


if __name__ == "__main__":
    unittest.main()
