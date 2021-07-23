import unittest
import numpy as np
import oneflow as flow
from scipy import special
from collections import OrderedDict
import oneflow as flow
from test_util import GenArgList


def _test_erf_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.erf(of_input)
    np_out = special.erf(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(
            of_input.grad.numpy(),
            2 / np.sqrt(np.pi) * np.exp(-np.square(of_input.numpy())),
            1e-05,
            1e-05,
        )
    )


def _test_tensor_erf_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input.erf()
    np_out = special.erf(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(
            of_input.grad.numpy(),
            2 / np.sqrt(np.pi) * np.exp(-np.square(of_input.numpy())),
            1e-05,
            1e-05,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestErfModule(flow.unittest.TestCase):
    def test_erf(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_erf_impl(test_case, *arg)
            _test_tensor_erf_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
