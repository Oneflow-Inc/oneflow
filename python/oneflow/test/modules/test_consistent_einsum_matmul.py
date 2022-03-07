import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=2, check_graph=False)
def _test_einsum_matmul(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    dim2 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=2, dim0=dim1, dim1=dim2,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    # NOTE(Liang Depeng): the same as 'ik,kj->ij'
    z = torch.einsum("ik,kj", g_x, g_y)
    return z

class TestEinsumConsistent(flow.unittest.TestCase):
    @globaltest
    def test_einsum_matmul(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_matmul(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
