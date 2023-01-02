"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
import random
import numpy as np
from collections import OrderedDict
import torch

import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList


def _test_multinomial(test_case, device, seed, replacement, dtype):
    n_dists = random.randint(8, 64)
    n_categories = random.randint(8, 64)
    num_samples = random.randint(4, n_categories)

    weights_torch = torch.rand(
        n_dists,
        n_categories,
        device=device,
        dtype=torch.float32 if dtype == "float" else torch.float64,
    )
    weights_oneflow = flow.tensor(
        weights_torch.cpu().numpy(),
        device=device,
        dtype=flow.float32 if dtype == "float" else flow.float64,
    )

    torch.manual_seed(seed)
    flow.manual_seed(seed)

    torch_res = torch.multinomial(
        weights_torch, num_samples, replacement=replacement, generator=None
    )
    flow_res = flow.multinomial(
        weights_oneflow, num_samples, replacement=replacement, generator=None
    )

    test_case.assertTrue(
        np.allclose(torch_res.cpu().numpy(), flow_res.cpu().numpy(), atol=1e-8,)
    )

    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)
    oneflow_gen = flow.Generator(device=device)
    oneflow_gen.manual_seed(seed)

    torch_res = torch.multinomial(
        weights_torch, num_samples, replacement=replacement, generator=torch_gen
    )
    flow_res = flow.multinomial(
        weights_oneflow, num_samples, replacement=replacement, generator=oneflow_gen
    )

    test_case.assertTrue(
        np.allclose(torch_res.cpu().numpy(), flow_res.cpu().numpy(), atol=1e-8,)
    )


@flow.unittest.skip_unless_1n1d()
class TestMultinomial(flow.unittest.TestCase):
    def test_multinomial(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["seed"] = [0, 2, 4]
        arg_dict["replacement"] = [True, False]
        arg_dict["dtype"] = ["double", "float"]
        for arg in GenArgList(arg_dict):
            _test_multinomial(test_case, *arg[0:])


if __name__ == "__main__":
    unittest.main()
