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
import os
import unittest
from collections import OrderedDict

import numpy as np
import torch as torch_original
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def do_test_rrelu_same_bound(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    flow.manual_seed(233)
    torch_original.manual_seed(233)

    flow_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    torch_tensor = torch_original.tensor(np_x, requires_grad=True, device=device)

    rate = np.random.randn()
    flow_rrelu = flow.nn.RReLU(lower=rate, upper=rate)
    torch_rrelu = torch_original.nn.RReLU(lower=rate, upper=rate)
    flow_out = flow_rrelu(flow_tensor)
    torch_out = torch_rrelu(torch_tensor)

    test_case.assertTrue(
        np.allclose(
            flow_out.cpu().detach().numpy(),
            torch_out.cpu().detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
    )
    flow_out.sum().backward()
    torch_out.sum().backward()
    test_case.assertTrue(
        np.allclose(
            flow_tensor.grad.cpu().detach().numpy(),
            torch_tensor.grad.cpu().detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
    )


def do_test_rrelu_different_bound(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    flow_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    rate = np.random.randn()
    flow_rrelu = flow.nn.RReLU(lower=rate, upper=rate + 0.5)
    flow_out = flow_rrelu(flow_tensor)
    flow_out.sum().backward()
    flow_grad = flow_tensor.grad
    flow_div = flow_out / flow_tensor
    test_case.assertTrue(
        np.allclose(
            (flow.where(flow_tensor >= 0, 1, 0)).cpu().detach().numpy(),
            (flow.where(flow_div == 1.0, 1, 0)).cpu().detach().numpy(),
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            (flow.where(flow_tensor < 0, 1, 0)).cpu().detach().numpy(),
            (
                flow.where(
                    flow.logical_and(
                        flow.logical_and(flow_div >= rate, flow_div <= (rate + 0.5)),
                        flow_tensor < 0,
                    ),
                    1,
                    0,
                )
            )
            .cpu()
            .detach()
            .numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            flow_grad.cpu().detach().numpy(),
            flow_div.cpu().detach().numpy(),
            rtol=1e-1,
            atol=1e-4,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_numpy_case(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            do_test_rrelu_same_bound,
            do_test_rrelu_different_bound,
        ]
        arg_dict["shape"] = [
            [20],
            [12, 32],
            [4, 47, 156],
            [5, 33, 65],
            [3, 132, 94],
            [9, 256, 63],
        ]
        # NOTE(hujiakui): in PyTorch <= 1.13, the CUDA RReLU Backward Function of PyTorch is wrong.
        if float(torch_original.__version__[:4]) < 1.13:
            arg_dict["device"] = ["cpu"]
        else:
            arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [np.float32, np.float64]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_functional_rrelu(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        lower = np.abs(
            np.random.randn()
        )  # In-place leakyReLu backward calculation is triggered with a negative slope which is not supported
        return torch.nn.functional.rrelu(
            x, lower=lower, upper=lower + 0.5, inplace=random_bool(), training=False,
        )

    @autotest(n=5)
    @unittest.skipIf(
        float(torch_original.__version__[:4]) < 1.13
        and not os.getenv("ONEFLOW_TEST_CPU_ONLY"),
        f"RReLU CUDA test need pytorch version >= 1.13, got {torch_original.__version__}",
    )
    def test_rrelu_train(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        lower = np.abs(np.random.randn())
        m = torch.nn.RReLU(lower=lower, upper=lower, inplace=random_bool())
        return m(x)

    @autotest(n=5, check_graph=False)
    def test_rrelu_eval(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        lower = np.abs(np.random.randn())
        m = torch.nn.RReLU(lower=lower, upper=lower, inplace=random_bool()).eval()
        return m(x)

    @profile(torch.nn.functional.rrelu)
    def profile_rrelu(test_case):
        lower = np.random.randn()
        torch.nn.functional.rrelu(
            torch.ones(1, 128, 28, 28),
            lower=lower,
            upper=lower + 0.5,
            inplace=False,
            training=True,
        )
        torch.nn.functional.rrelu(
            torch.ones(1, 128, 28, 28),
            lower=lower,
            upper=lower + 0.5,
            inplace=True,
            training=True,
        )

        torch.nn.functional.rrelu(
            torch.ones(16, 128, 28, 28),
            lower=lower,
            upper=lower + 0.5,
            inplace=False,
            training=True,
        )

        torch.nn.functional.rrelu(
            torch.ones(16, 128, 28, 28),
            lower=lower,
            upper=lower + 0.5,
            inplace=True,
            training=True,
        )


if __name__ == "__main__":
    unittest.main()
