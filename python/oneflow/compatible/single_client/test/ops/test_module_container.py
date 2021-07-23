import unittest
from typing import Tuple
from oneflow.compatible.single_client import experimental as flow
from oneflow.compatible.single_client import typing as tp

@unittest.skipIf(not flow.unittest.env.eager_execution_enabled(), "module doesn't work in lazy mode now")
class TestContainer(flow.unittest.TestCase):

    def test_module_forward(test_case):

        class CustomModule(flow.nn.Module):

            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x):
                return x + self.w
        m1 = CustomModule(5)
        m2 = CustomModule(4)
        s = flow.nn.Sequential(m1, m2)
        test_case.assertEqual(s(1), 10)
if __name__ == '__main__':
    unittest.main()