import oneflow as flow
from oneflow.fx import symbolic_trace, replace_pattern
from oneflow.test_utils.automated_test_util import *
import unittest

class M(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        val1 = flow.neg(w1)
        m1 = flow.cat([val1, w2]).sum()
        val2 = flow.neg(w1)
        m2 = flow.cat([val2, w2]).sum()
        return x + flow.max(m1) + flow.max(m2)
    
class TestReplaceOps(flow.unittest.TestCase):
    def test_pattern(test_case):
        traced = symbolic_trace(M())
        def pattern(a1, a2):
            val1 = flow.neg(a1)
            return flow.cat([val1, a2]).sum()

        def replacement(w1, w2):
            return flow.stack([w1, w2])

        replace_pattern(traced, pattern, replacement)

        test_case.assertTrue('cat' not in traced.code)
        test_case.assertTrue('neg' not in traced.code)
        test_case.assertTrue('stack' in traced.code)

if __name__ == '__main__':
    unittest.main()