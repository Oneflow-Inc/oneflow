import unittest
import oneflow as flow
from automated_test_util import *

@flow.unittest.skip_unless_1n1d()
class TestSoftmaxModule(flow.unittest.TestCase):
    @autotest()
    def test_against_pytorch(test_case):
        dim = 2
        m = torch.nn.LogSoftmax(dim=1)
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=dim,dim1 = random(2,3),dim2 = random(2,3)).to(device)
        y = m(x)
        return y

if __name__ == "__main__":
    unittest.main()
