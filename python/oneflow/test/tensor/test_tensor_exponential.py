import unittest
import random
import numpy as np
from collections import OrderedDict
import torch

import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

def _test_exponential(test_case, device, seed, lambd, dtype):
    torch.manual_seed(seed)
    flow.manual_seed(seed)

    dim1 = random.randint(8, 64)
    dim2 = random.randint(8, 64)

    torch_arr = torch.zeros(dim1, device=device, dtype=torch.float32 if dtype == "float" else torch.float64).exponential_(lambd=lambd, generator=None)
    oneflow_arr = flow.zeros(dim1, device=device, dtype=flow.float32 if dtype == "float" else flow.float64).exponential_(lambd=lambd, generator=None)

    test_case.assertTrue(
        np.allclose(
            torch_arr.cpu().numpy(),
            oneflow_arr.cpu().numpy(),
            atol=1e-8,
        )
    )

    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)
    torch_arr = torch.zeros(dim1, device=device, dtype=torch.float32 if dtype == "float" else torch.float64).exponential_(lambd=lambd, generator=torch_gen)
    oneflow_gen = flow.Generator(device=device)
    oneflow_gen.manual_seed(seed)
    oneflow_arr = flow.zeros(dim1, device=device, dtype=flow.float32 if dtype == "float" else flow.float64).exponential_(lambd=lambd, generator=oneflow_gen)

    test_case.assertTrue(
        np.allclose(
            torch_arr.cpu().numpy(),
            oneflow_arr.cpu().numpy(),
            atol=1e-8,
        )
    )

@flow.unittest.skip_unless_1n1d()
class TestExponential(flow.unittest.TestCase):
    def test_exponential(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["seed"] = [0, 2, 4]
        arg_dict["lambd"] = [1, 0.5, 0.1]
        arg_dict["dtype"] = ["double", "float"]
        for arg in GenArgList(arg_dict):
            _test_exponential(test_case, *arg[0:])


if __name__ == "__main__":
    unittest.main()
