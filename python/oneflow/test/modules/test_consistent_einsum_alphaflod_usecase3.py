import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=2, check_graph=False)
def _test_einsum_alphaflod_usecase3(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(ndim=2, dim0=dim0, dim1=dim1,)
    y = random_tensor(ndim=3, dim0=dim0, dim1=dim1, dim2=random(1, 3) * 8,)
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("ra,rab->rb", g_x, g_y)
    return z

class TestEinsumConsistent(flow.unittest.TestCase):
    @globaltest
    def test_einsum_alphaflod_usecase3(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_einsum_alphaflod_usecase3(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
