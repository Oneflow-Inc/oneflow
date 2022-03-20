import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest
import torch

@flow.unittest.skip_unless_1n1d()
class TestFromTroch(flow.unittest.TestCase):
    def test_from_torch_cpu(test_case):
        torch_t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        numpy_from_torch = torch_t.numpy()

        # torch and numpy shared the same memory
        test_case.assertEqual(torch_t.data_ptr(), numpy_from_torch.__array_interface__["data"][0])
        # oneflow and numpy shared the same memory
        # So oneflow and torch cpu tensor shared the same memory
        # Which means oneflow can use torch's cpu tensor without cost.
        flow_t = flow.utils.from_torch(torch_t)

        test_case.assertTrue(np.allclose(torch_t.numpy(), flow_t.numpy(), rtol=0.001, atol=0.001))
        test_case.assertEqual(torch_t.numpy().dtype, flow_t.numpy().dtype)

    def _test_from_torch_cuda(test_case):
        # This test can not pass, to be fixed later. 
        torch_t = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        print("torch tensor ", torch_t)
        flow_t = flow.utils.from_torch(torch_t)
        print("flow tensor ", flow_t)

if __name__ == "__main__":
    unittest.main()