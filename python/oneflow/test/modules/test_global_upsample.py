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
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_global_upsample2d_nearest(test_case, placement, sbp):
    x = random_tensor(ndim=3, dim0=8, dim1=16).to_global(placement, sbp)
    print(x)
    m = torch.nn.Upsample(scale_factor=random().to(int), mode="nearest",)
    y = m(x)
    return y


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_global_upsample2d_linear(test_case, placement, sbp):
    x = random_tensor(ndim=3, dim0=8, dim1=16).to_global(placement, sbp)
    m = torch.nn.Upsample(
        scale_factor=random().to(int), mode="linear", align_corners=random_bool(),
    )
    y = m(x)
    return y


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_global_upsample2d_bilinear(test_case, placement, sbp):
    x = random_tensor(ndim=4, dim0=8, dim1=16).to_global(placement, sbp)
    m = torch.nn.Upsample(
        scale_factor=random().to(int), mode="bilinear", align_corners=random_bool(),
    )
    y = m(x)
    return y


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_global_upsample2d_bicubic(test_case, placement, sbp):
    x = random_tensor(ndim=4, dim0=8, dim1=16).to_global(placement, sbp)
    m = torch.nn.Upsample(
        scale_factor=random().to(int), mode="bicubic", align_corners=random_bool(),
    )
    y = m(x)
    return y


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_global_upsample2d_trilinear(test_case, placement, sbp):
    x = random_tensor(ndim=5, dim0=8, dim1=16).to_global(placement, sbp)
    m = torch.nn.Upsample(
        scale_factor=random().to(int), mode="trilinear", align_corners=random_bool(),
    )
    y = m(x)
    return y


class TestGlobalUpsample2d(flow.unittest.TestCase):
    @unittest.skip(
        "The nearest interpolate operation in pytorch has bug, https://github.com/pytorch/pytorch/issues/65200"
    )
    @globaltest
    def test_global_upsample2d_nearest(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_upsample2d_nearest(test_case, placement, sbp)

    @globaltest
    def test_global_upsample2d_linear(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_upsample2d_linear(test_case, placement, sbp)

    @globaltest
    def test_global_upsample2d_bilinear(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_upsample2d_bilinear(test_case, placement, sbp)

    @globaltest
    def test_global_upsample2d_bicubic(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_upsample2d_bicubic(test_case, placement, sbp)

    @globaltest
    def test_global_upsample2d_trilinear(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_upsample2d_trilinear(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
