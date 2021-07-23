import random
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow


def _test_stack(test_case, device, shape):
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)
    x_tensor = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y_tensor = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    out_np = np.stack([x, y], axis=1)
    out_of = flow.stack([x_tensor, y_tensor], dim=1).numpy()
    test_case.assertTrue(np.allclose(out_np, out_of, 1e-05, 1e-05))


def _test_stack_tuple_input(test_case, device, shape):
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)
    x_tensor = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    y_tensor = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))
    out_np = np.stack([x, y], axis=0)
    out_of = flow.stack((x_tensor, y_tensor), dim=0).numpy()
    test_case.assertTrue(np.allclose(out_np, out_of, 1e-05, 1e-05))


def _test_stack_backward(test_case, device, shape):
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)
    x_tensor = flow.Tensor(x, device=flow.device(device), requires_grad=True)
    y_tensor = flow.Tensor(y, device=flow.device(device), requires_grad=True)
    out_of = flow.stack([x_tensor, y_tensor]).sum()
    out_of.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np.ones(x_tensor.shape), 1e-05, 1e-05)
    )
    test_case.assertTrue(
        np.allclose(y_tensor.grad.numpy(), np.ones(y_tensor.shape), 1e-05, 1e-05)
    )


def _test_stack_different_dim(test_case, device, shape):
    x = np.random.rand(*shape)
    y = np.random.rand(*shape)
    x_tensor = flow.Tensor(x, device=flow.device(device))
    y_tensor = flow.Tensor(y, device=flow.device(device))
    for axis in range(-len(x.shape) - 1, len(x.shape) + 1):
        out_of = flow.stack([x_tensor, y_tensor], dim=axis)
        out_np = np.stack([x, y], axis=axis)
        test_case.assertTrue(np.allclose(out_np, out_of.numpy(), 1e-05, 1e-05))


def _test_stack_multi_input(test_case, device, shape):
    max_input_num = 10
    for i in range(2, max_input_num):
        x = []
        x_tensor = []
        for _ in range(0, i):
            tmp = np.random.rand(*shape)
            x.append(tmp)
            x_tensor.append(flow.Tensor(tmp, device=flow.device(device)))
        out_of = flow.stack(x_tensor, dim=-1)
        out_np = np.stack(x, axis=-1)
        test_case.assertTrue(np.allclose(out_np, out_of.numpy(), 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestStack(flow.unittest.TestCase):
    def test_stack(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_stack,
            _test_stack_tuple_input,
            _test_stack_backward,
            _test_stack_different_dim,
            _test_stack_multi_input,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [
            tuple((random.randrange(1, 10) for _ in range(i))) for i in range(3, 6)
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
