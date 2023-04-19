import os
import unittest


class MLIRTestCase(unittest.TestCase):
    def tearDown(self):
        for key in os.environ.keys():
            if key.startswith("ONEFLOW_MLIR"):
                os.environ.pop(key)
