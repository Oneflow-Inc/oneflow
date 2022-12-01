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
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestBatchNormModule(flow.unittest.TestCase):
    @autotest(
        auto_backward=True, rtol=1e-3, atol=1e-3, check_grad_use_random_data=False
    )
    def test_batchnorm1d_module_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm1d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        x = random_tensor(
            ndim=3, dim0=random(1, 4), dim1=channel, requires_grad=True
        ).to(device)
        y = m(x)
        return y

    @autotest(
        auto_backward=True, rtol=1e-3, atol=1e-3, check_grad_use_random_data=False
    )
    def test_batchnorm2d_module_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm2d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        x = random_tensor(
            ndim=4, dim0=random(1, 4), dim1=channel, requires_grad=True
        ).to(device)
        y = m(x)
        return y

    @autotest(
        auto_backward=True, rtol=1e-3, atol=1e-3, check_grad_use_random_data=False
    )
    def test_batchnorm3d_module_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm3d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        x = random_tensor(ndim=5, dim1=channel, requires_grad=True).to(device)
        y = m(x)
        return y

    @autotest(rtol=1e-3, atol=1e-3, check_grad_use_random_data=False)
    def test_functional_batchnorm_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        x = random_tensor(ndim=5, dim1=channel, requires_grad=True).to(device)
        running_mean = random_tensor(ndim=1, dim0=channel, requires_grad=False)
        running_var = random_tensor(ndim=1, dim0=channel, low=0.0, requires_grad=False)
        weight = random_tensor(ndim=1, dim0=channel)
        bias = random_tensor(ndim=1, dim0=channel)
        result = torch.nn.functional.batch_norm(
            input=x,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=random_bool(),
        )
        return result

    @autotest(rtol=1e-3, atol=1e-3, auto_backward=False, check_graph=False)
    def test_batchnorm2d_module_with_half_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm2d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        m.half()
        x = random_tensor(
            ndim=4, dim0=random(1, 4), dim1=channel, requires_grad=True
        ).to(device)
        x.half()
        y = m(x)
        return y

    @profile(torch.nn.functional.batch_norm)
    def profile_batchnorm(test_case):
        input = torch.ones(16, 128, 28, 28)
        running_mean = torch.randn(128)
        running_var = torch.randn(128)
        weight = torch.randn(128)
        bias = torch.randn(128)
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, weight, bias, True
        )
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, weight, bias, False
        )
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, None, None, True
        )
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, None, None, False
        )


if __name__ == "__main__":
    unittest.main()
