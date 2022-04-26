import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_logical_slice_assign(test_case, placement, sbp):
    x = random_tensor(2, 4, 4, requires_grad=False).oneflow
    x_numpy = x.detach().cpu().numpy()

    x = x.to_global(placement=placement, sbp=sbp)
    x[:, :2] = 3 
    x_numpy[:, :2] = 3

    test_case.assertTrue(x.sbp in [(oneflow.sbp.broadcast,)])
    test_case.assertTrue(np.array_equal(x.numpy(), x_numpy))

def _test_graph_logical_slice_assign(test_case, placement, sbp):
    x = random_tensor(2, 4, 4, requires_grad=False).oneflow
    x_numpy = x.detach().cpu().numpy()

    class SliceAssignGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
        
        def build(self, x):
            x[:, :2] = 3 
            return x
    
    slice_assing_g = SliceAssignGraph()
    slice_assing_g.debug(2)

    x = x.to_global(placement=placement, sbp=sbp)

    y = slice_assing_g(x)

    x_numpy[:, :2] = 3

    test_case.assertTrue(y.sbp in [(oneflow.sbp.broadcast,)])
    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy))


class TestGlobalLogicalSliceAssign(flow.unittest.TestCase):
    @globaltest
    def test_logical_slice_assign(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                # logical slice assign not support 2d sbp currently
                # logical slice assign only support broadcast currently
                if len(sbp) > 1 or sbp[0] != flow.sbp.broadcast:
                    continue
                _test_logical_slice_assign(test_case, placement, sbp)
                _test_graph_logical_slice_assign(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
