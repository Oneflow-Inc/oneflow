import unittest
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_less_normal(test_case, device):
    input1 = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.allclose(input1, input2)
    np_out = np.allclose(input1.numpy(), input2.numpy())
    test_case.assertTrue(of_out == np_out)


def _test_less_symbol(test_case, device):
    input1 = flow.tensor(
        np.array([1, 1, 4]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    input2 = flow.tensor(
        np.array([1, 2, 3]).astype(np.float32),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.allclose(input1, input2)
    np_out = np.allclose(input1.numpy(), input2.numpy())
    test_case.assertTrue(of_out == np_out)


@flow.unittest.skip_unless_1n1d()
class TestLess(flow.unittest.TestCase):
    def test_less(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_less_normal, _test_less_symbol]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_less_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = torch.allclose(x1, oneof(x2, random().to(int).to(float)))
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_less_with_0dim_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(ndim=0).to(device)
        x2 = random_tensor(ndim=0).to(device)
        y = torch.allclose(x1, oneof(x2, random().to(int).to(float)))
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_tensor_less_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = x1.allclose(oneof(x2, random().to(int), random().to(float)))
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_less_bool_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        x2 = random_tensor(len(shape), *shape, requires_grad=False).to(
            device=device, dtype=torch.bool
        )
        y = torch.allclose(x1, oneof(x2, random().to(int).to(float)))
        return y

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_tensor_less_with_0dim_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x1 = random_tensor(ndim=0).to(device)
        x2 = random_tensor(ndim=0).to(device)
        y = x1.allclose(oneof(x2, random().to(int), random().to(float)))
        return y


if __name__ == "__main__":
    unittest.main()
