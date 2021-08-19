import unittest
from automated_test_util import *
import oneflow as flow
import oneflow.unittest



@flow.unittest.skip_unless_1n1d()
class TestReduceProd(flow.unittest.TestCase):
    @autotest()
    def test_reduce_prod_without_dim(test_case):
        device = random_device()

        ndim = random(1, 5).to(int)
        x = random_pytorch_tensor(ndim=ndim).to(device)
        y = torch.prod(x)

        return y

    @autotest()
    def test_reduce_prod_with_dim(test_case):
        device = random_device()

        ndim = random(1, 5).to(int)
        x = random_pytorch_tensor(ndim=ndim).to(device)

        dim = random(0, ndim).to(int)

        y = torch.prod(x, dim)

        return y


if __name__ == "__main__":
    unittest.main()
