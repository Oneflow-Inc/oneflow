
import argparse
import numpy as np
import os
import time
import unittest

import oneflow as flow
import oneflow.unittest

@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestBlockGetDtype(oneflow.unittest.TestCase):
    def test_block_get_dtype(test_case):
        class ModuleDtype(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = flow.float64
                self.linear = flow.nn.Linear(3, 8, False).to("cuda")
                print(self.linear.weight.dtype)
            
            def forward(self, x):
                dt = self.dtype
                print(dt)
                self.linear.to(dtype=self.dtype)
                print(self.linear.weight.dtype)
                return self.linear(x)
        
        m = ModuleDtype()

        np_in = np.ones((8, 3)).astype(np.float64)
        x = flow.tensor(np_in, device="cuda")

        @flow.nn.Graph.to_graph
        def block_get_dtype(x):
            return m(x)
        
        out = block_get_dtype(x)
        

if __name__ == "__main__":
    unittest.main()