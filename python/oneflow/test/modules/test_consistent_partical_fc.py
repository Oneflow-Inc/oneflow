import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest

from test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

class TestParitalFC(flow.unittest.TestCase):
    @globaltest
    def test_parital_fc(test_case):
        placement = flow.env.all_device_placement("cuda")
        w =  flow.randn(5000, 128, placement=placement, sbp=flow.sbp.split(0))
        label = flow.randint(0, 5000, (512,), placement=placement, sbp=flow.sbp.split(0))
        num_sample = 500
        out = flow.distributed_partial_fc_sample(w, label, num_sample)
        
        w =  flow.randn(5000, 128, placement=placement, sbp=flow.sbp.broadcast)
        label = flow.randint(0, 5000, (512,), placement=placement, sbp=flow.sbp.split(0))
        num_sample = 500
        out = flow.distributed_partial_fc_sample(w, label, num_sample)
        test_case.assertTrue(out[0].shape == flow.Size([512]))
        test_case.assertTrue(out[1].shape == flow.Size([500]))
        test_case.assertTrue(out[2].shape == flow.Size([500, 128]))

if __name__ == "__main__":
    unittest.main()