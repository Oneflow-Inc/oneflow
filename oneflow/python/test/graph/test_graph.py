import unittest

import numpy as np

import oneflow.experimental as flow

@flow.unittest.skip_unless_1n1d()
class TestGraph(flow.unittest.TestCase):
    def test_nested_module(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = flow.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        def np_relu(np_arr):
            return np.where(np_arr > 0, np_arr, 0)

        m = CustomModule()
        x = flow.Tensor(2, 3)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        y = m(x)

        test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))