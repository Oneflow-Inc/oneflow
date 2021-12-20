import os
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestModuleToCosistent(flow.unittest.TestCase):
    def test_module_to_consistent(test_case):
	rank = flow.env.get_rank()
	P = flow.placement("cuda", {0:[0, 1]})
	B = flow.sbp.broadcast

        class ReuseVarModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(3, 4)
                self.linear2 = flow.nn.Linear(3, 4)
                self.linear2.weight = self.linear1.weight
        
        reuse_var_m = ReuseVarModule()

        test_case.assertTrue(reuse_var_m.linear1.weight is reuse_var_m.linear2.weight)
        test_case.assertEqual(reuse_var_m.linear1.weight.device, flow.device("cpu", rank))

        test_case.assertTrue(reuse_var_m.linear1.bias is not reuse_var_m.linear2.bias)
        test_case.assertEqual(reuse_var_m.linear1.bias.device, flow.device("cpu", rank))

        reuse_var_m.to_consistent(placement=P, sbp=B)

        test_case.assertTrue(reuse_var_m.linear1.weight is reuse_var_m.lineare.weight)
        test_case.assertEqual(reuse_var_m.linear1.weight.placement, P)
        test_case.assertEqual(reuse_var_m.linear1.weight.sbp, B)

        test_case.assertTrue(reuse_var_m.linear1.bias is not reuse_var_m.linear2.bias)
        test_case.assertEqual(reuse_var_m.linear1.bias.placement, P)
        test_case.assertEqual(reuse_var_m.linear1.bias.sbp, B)
            

if __name__ == "__main__":
    unittest.main()