import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow


def _test_reciprocal_impl(test_case, shape, device):
    x = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.reciprocal(x)
    np_out = np.reciprocal(x.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestReciprocalModule(flow.unittest.TestCase):
    def test_reciprocal(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_reciprocal_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
