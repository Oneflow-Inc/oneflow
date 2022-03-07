import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=2, check_graph=False)
def _test_einsum_tensor_contraction(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(
        ndim=4, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1, dim3=random(1, 3) * 8,
    )
    y = random_tensor(
        ndim=5,
        dim0=random(1, 3) * 8,
        dim1=random(1, 3) * 8,
        dim2=dim0,
        dim3=random(1, 3) * 8,
        dim4=dim1,
    )
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("pqrs,tuqvr->pstuv", g_x, g_y)
    return z

class TestEinsumConsistent(flow.unittest.TestCase):
    @globaltest
    def test_einsum_tensor_contraction(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_einsum_tensor_contraction(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
