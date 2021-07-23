import unittest
import numpy as np
import oneflow as flow

@flow.unittest.skip_unless_1n1d()
class TestTo(flow.unittest.TestCase):

    def test_tensor_to_h2d(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5))
        output = input.to(device=flow.device('cuda'))
        test_case.assertEqual(output.device, flow.device('cuda'))
        test_case.assertTrue(np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001))
        gpu_output = output.to(device=flow.device('cuda'))
        test_case.assertEqual(gpu_output.device, flow.device('cuda'))
        test_case.assertTrue(np.allclose(input.numpy(), gpu_output.numpy(), rtol=0.0001, atol=0.0001))

    def test_tensor_to_d2h(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5), device=flow.device('cuda'))
        output = input.to(device=flow.device('cpu'))
        test_case.assertEqual(output.device, flow.device('cpu'))
        test_case.assertTrue(np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001))

    def test_tensor_to_d2d(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5), device=flow.device('cuda'))
        output = input.to(device=flow.device('cuda:0'))
        test_case.assertEqual(output.device, flow.device('cuda:0'))
        test_case.assertTrue(np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001))

    def test_tensor_to_h2h(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5))
        output = input.to(device=flow.device('cpu'))
        test_case.assertEqual(output.device, flow.device('cpu'))
        test_case.assertTrue(np.allclose(input.numpy(), output.numpy(), rtol=0.0001, atol=0.0001))

    def test_tensor_to_cast(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5))
        output = input.to(dtype=flow.int)
        test_case.assertEqual(output.dtype, flow.int)

    def test_tensor_to_cast_h2d(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5))
        output = input.to(device=flow.device('cuda'), dtype=flow.int)
        test_case.assertEqual(output.dtype, flow.int)
        test_case.assertEqual(output.device, flow.device('cuda'))

    def test_tensor_using_tensor(test_case):
        tensor = flow.Tensor(np.random.randn(2, 3, 4, 5), device='cuda', dtype=flow.int)
        input = flow.Tensor(np.random.randn(2, 3))
        output = input.to(tensor)
        test_case.assertEqual(output.dtype, flow.int)
        test_case.assertEqual(output.device, flow.device('cuda'))
if __name__ == '__main__':
    unittest.main()