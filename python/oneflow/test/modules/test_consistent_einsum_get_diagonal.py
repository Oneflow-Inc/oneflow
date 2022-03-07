import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=2, check_graph=False)
def _test_einsum_get_diagonal(test_case, placement, sbp):
    dim = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim, dim1=dim,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ii->i", g_x)
    return z

class TestEinsumConsistent(flow.unittest.TestCase):
    @globaltest
    def test_einsum_get_diagonal(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_get_diagonal(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
