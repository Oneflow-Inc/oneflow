import unittest
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest

rtol = 1e-3


def _perturbate(x):
    shape = x.oneflow.shape
    device = x.device
    diff = (
        random_tensor(len(shape), *shape, low=-1, high=1, requires_grad=False).to(
            device
        )
        * rtol
        * 2
    )
    return x + diff


@flow.unittest.skip_unless_1n1d()
class TestAllClose(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_allclose_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(requires_grad=False).to(device)
        x2 = _perturbate(x1)
        y = torch.allclose(x1, x2, rtol=rtol)
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_allclose_with_0dim_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(requires_grad=False).to(device)
        x2 = _perturbate(x1)
        y = torch.allclose(x1, x2, rtol=rtol)
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_tensor_allclose_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(requires_grad=False).to(device)
        x2 = _perturbate(x1)
        y = x1.allclose(x2, rtol=rtol)
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_allclose_broadcast(test_case):
        device = random_device()
        x1 = random_tensor(2, 2, 8, requires_grad=False).to(device)
        x2 = _perturbate(x1[:, :1])
        y = torch.allclose(x1, x2, rtol=rtol)
        return y


if __name__ == "__main__":
    unittest.main()
