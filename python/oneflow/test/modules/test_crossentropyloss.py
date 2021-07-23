import unittest
import numpy as np
import oneflow as flow
from automated_test_util import *

@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyLossModule(flow.unittest.TestCase):

    @unittest.skip('nn.CrossEntropyLoss has bug')
    @autotest(n=200)
    def test_CrossEntropyLoss_with_random_data(test_case):
        num_classes = random()
        shape = random_tensor(ndim=random(2, 5), dim1=num_classes).value().shape
        m = torch.nn.CrossEntropyLoss(reduction=oneof('none', 'sum', 'mean', nothing()), ignore_index=random(0, num_classes) | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(len(shape), *shape).to(device)
        target = random_pytorch_tensor(len(shape) - 1, *shape[:1] + shape[2:], low=0, high=num_classes, dtype=int).to(device)
        y = m(x, target)
        return y
if __name__ == '__main__':
    unittest.main()