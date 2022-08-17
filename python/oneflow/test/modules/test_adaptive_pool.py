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

import oneflow as flow
import oneflow.unittest
from oneflow.nn.common_types import _size_1_t
from packaging import version
import torch as torch_original
from typing import Union, Tuple


from oneflow.test_utils.automated_test_util import *

NoneType = type(None)
# Not the same as those in PyTorch because 'output_size' cannot be NoneType (even in 'torch.nn.AdaptiveAvgPoolXd')
_size_2_opt_t_not_none = Union[int, Tuple[Union[int, NoneType], Union[int, NoneType]]]
_size_3_opt_t_not_none = Union[
    int, Tuple[Union[int, NoneType], Union[int, NoneType], Union[int, NoneType]]
]


@flow.unittest.skip_unless_1n1d()
class TestAdaptiveAvgPool(flow.unittest.TestCase):
    @autotest(n=5)
    def test_adaptive_avgpool1d(test_case):
        m = torch.nn.AdaptiveAvgPool1d(output_size=random().to(_size_1_t))
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3).to(device)
        y = m(x)
        return y

    @profile(torch.nn.functional.adaptive_avg_pool1d)
    def profile_adaptive_avg_pool1d(test_case):
        torch.nn.functional.adaptive_avg_pool1d(torch.ones(1, 64, 8), 5)

    @autotest(n=5)
    def test_adaptive_avgpool2d(test_case):
        m = torch.nn.AdaptiveAvgPool2d(output_size=random().to(_size_2_opt_t_not_none))
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4).to(device)
        y = m(x)
        return y

    @profile(torch.nn.functional.adaptive_avg_pool2d)
    def profile_adaptive_avg_pool2d(test_case):
        torch.nn.functional.adaptive_avg_pool2d(torch.ones(1, 64, 10, 9), 7)
        torch.nn.functional.adaptive_avg_pool2d(torch.ones(1, 64, 8, 9), (5, 7))

    @unittest.skipIf(
        version.parse(torch_original.__version__) < version.parse("1.10.0"),
        "GPU version 'nn.AdaptiveAvgPool3d' has a bug in PyTorch before '1.10.0'",
    )
    @autotest(n=5)
    def test_adaptive_avgpool3d(test_case):
        m = torch.nn.AdaptiveAvgPool3d(output_size=random().to(_size_3_opt_t_not_none))
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=5).to(device)
        y = m(x)
        return y

    @profile(torch.nn.functional.adaptive_avg_pool3d)
    def profile_adaptive_avg_pool3d(test_case):
        torch.nn.functional.adaptive_avg_pool3d(torch.ones(1, 64, 8, 9, 10), (5, 7, 9))
        torch.nn.functional.adaptive_avg_pool3d(torch.ones(1, 64, 10, 9, 8), 7)


@flow.unittest.skip_unless_1n1d()
class TestAdaptiveAvgPoolFunctional(flow.unittest.TestCase):
    @autotest(n=5)
    def test_adaptive_avgpool1d_functional(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        return torch.nn.functional.adaptive_avg_pool1d(x, output_size=random().to(int))

    @autotest(n=5)
    def test_adaptive_avgpool2d_functional(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        return torch.nn.functional.adaptive_avg_pool2d(x, output_size=random().to(int))

    @unittest.skipIf(
        version.parse(torch_original.__version__) <= version.parse("1.10.0"),
        "GPU version 'nn.AdaptiveAvgPool3d' has a bug in PyTorch before '1.10.0'",
    )
    @autotest(n=5)
    def test_adaptive_avgpool3d_functional(test_case):
        device = random_device()
        x = random_tensor(ndim=5).to(device)
        return torch.nn.functional.adaptive_avg_pool3d(x, output_size=random().to(int))


if __name__ == "__main__":
    unittest.main()
