import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *

@autotest(n=1, check_graph=False)
def _test_triu(test_case, placement, sbp):

    x = random_pytorch_tensor(2, 8, 8)
   
    x = x.to_consistent(placement=placement, sbp=sbp)
    z = torch.triu(x)
    return z

class TestTriuConsistent(flow.unittest.TestCase):
    @consistent
    def test_triu(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, dim=2):
                _test_triu(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
