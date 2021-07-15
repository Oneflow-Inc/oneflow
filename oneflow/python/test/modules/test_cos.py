import unittest
from collections import OrderedDict

import numpy as np
from scipy import special

import oneflow.experimental as flow
from test_util import GenArgList
from automated_test_util import *


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCos(flow.unittest.TestCase):
    def test_flow_cos_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            test_flow_against_pytorch(
                test_case, "cos", device=device,
            )

    def test_tensor_flow_cos_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            test_tensor_against_pytorch(
                test_case, "cos", device=device,
            )



if __name__ == "__main__":
    unittest.main()