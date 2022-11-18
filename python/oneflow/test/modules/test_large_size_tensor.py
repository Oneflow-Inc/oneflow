import random as random_util
import unittest

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
import numpy as np

@flow.unittest.skip_unless_1n1d()
class TestLargeSizeTensor(flow.unittest.TestCase):
    @autotest(n=1000, check_graph=False)
    def test(test_case):
        # size = random(2000, 3000)
        # size = 5000
        # x = random_tensor(ndim=1,dim0=size).cuda().half().requires_grad_()
        # y = random_tensor(ndim=1,dim0=size).cuda().half().requires_grad_()
        # z = x + y
        # weight = torch.randn_like(z)
        # p = z * weight
        # p.sum().backward()
        # import ipdb; ipdb.set_trace()
        # of_x = x.oneflow.grad.numpy()
        # torch_x = x.pytorch.grad.numpy()
        # diff = of_x - torch_x
        # return x + y
        size = random(200, 300)
        # x = random tensor(ndim=3， dim2=size).to("cuda").to(torch.half)
        # y = random tensor(ndim=3， dim2=size).to("cuda").to(torch.half)
        x = torch.Tensor(np.load("np_x.npy")).to("cuda").to(torch.half).requires_grad_()
        y = torch.Tensor(np.load("np_y.npy")).to("cuda").to(torch.half).requires_grad_()
        # np x = x.oneflow.numpy()# np_y = y.oneflow.numpy()
        # np.save("np x.npy",np x)
        # np.save("np_y.npy"，np_y)
        return x + y



if __name__ == "__main__":
    unittest.main()
