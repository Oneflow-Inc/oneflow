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
from packaging import version
import unittest
from typing import Union, Tuple
import torch as torch_original

import oneflow as flow
import oneflow.unittest
from oneflow.nn.common_types import _size_1_t
from oneflow.test_utils.automated_test_util import *

NoneType = type(None)
# Not the same as those in PyTorch because 'output_size' cannot be NoneType (even in 'torch.nn.AdaptiveAvgPoolXd')
_size_2_opt_t_not_none = Union[int, Tuple[Union[int, NoneType], Union[int, NoneType]]]
_size_3_opt_t_not_none = Union[
    int, Tuple[Union[int, NoneType], Union[int, NoneType], Union[int, NoneType]]
]


@autotest(n=1, check_graph=True)
def _test_adaptive_avgpoolnd(test_case, ndim, pool_size, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    if pool_size == 1:
        m = torch.nn.AdaptiveAvgPool1d(output_size=random().to(_size_1_t))
    elif pool_size == 2:
        m = torch.nn.AdaptiveAvgPool2d(output_size=random().to(_size_2_opt_t_not_none))
    elif pool_size == 3:
        m = torch.nn.AdaptiveAvgPool3d(output_size=random().to(_size_3_opt_t_not_none))
    else:
        raise ValueError("pool size should be 1, 2 or 3, but got %d" % pool_size)
    m.train(random())
    y = m(x)
    return y


@autotest(n=1, check_graph=True)
def _test_adaptive_avgpoolnd_functional(test_case, ndim, pool_size, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    if pool_size == 1:
        return torch.nn.functional.adaptive_avg_pool1d(x, output_size=random().to(int))
    elif pool_size == 2:
        return torch.nn.functional.adaptive_avg_pool2d(x, output_size=random().to(int))
    elif pool_size == 3:
        return torch.nn.functional.adaptive_avg_pool3d(x, output_size=random().to(int))


class TestAdaptiveAvgPool(flow.unittest.TestCase):
    @globaltest
    def test_adaptive_avgpool(test_case):
        for placement in all_placement():
            ndim = 3
            for sbp in all_sbp(placement, max_dim=2):
                _test_adaptive_avgpoolnd(test_case, ndim, 1, placement, sbp)
                _test_adaptive_avgpoolnd_functional(test_case, ndim, 1, placement, sbp)

            ndim = 4
            for sbp in all_sbp(placement, max_dim=2):
                _test_adaptive_avgpoolnd(test_case, ndim, 2, placement, sbp)
                _test_adaptive_avgpoolnd_functional(test_case, ndim, 2, placement, sbp)

            # GPU version 'nn.AdaptiveAvgPool3d' has a bug in PyTorch before '1.10.0'
            if (
                version.parse(torch_original.__version__) < version.parse("1.10.0")
                and placement.type == "cuda"
            ):
                continue
            ndim = 5
            for sbp in all_sbp(placement, max_dim=2):
                _test_adaptive_avgpoolnd(test_case, ndim, 3, placement, sbp)
                _test_adaptive_avgpoolnd_functional(test_case, ndim, 3, placement, sbp)


if __name__ == "__main__":
    unittest.main()
