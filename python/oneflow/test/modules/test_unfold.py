import unittest

import oneflow as flow
import oneflow.unittest
from automated_test_util import *
from oneflow.nn.common_types import _size_2_t


@flow.unittest.skip_unless_1n1d()
class TestUnfold(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=False, rtol=1e-3, atol=1e-3)
    def test_unfold_with_random_data(test_case):
        m = torch.nn.Unfold(
            kernel_size=random(1, 3).to(_size_2_t),
            dilation=random(1, 4).to(_size_2_t) | nothing(),
            padding=random(1, 3).to(_size_2_t) | nothing(),
            stride=random(1, 3).to(_size_2_t) | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=4, dim0=random(1, 3),dim1=random(2, 10), dim2=random(10, 20), dim3=random(10, 20)).to(device)
        y = m(x)
        return y

if __name__ == "__main__":
    unittest.main()
