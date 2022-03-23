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

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False)
def _test_clamp_flow_with_random_data(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clamp(input, min=random().to(float), max=random().to(float))
    return y


@autotest(n=1, check_graph=False)
def _test_clamp_min_none_flow_with_random_data(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clamp(input, max=random().to(float))
    return y


@autotest(n=1, check_graph=False)
def _test_clamp_max_none_flow_with_random_data(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clamp(input, min=random().to(float))
    return y


@autotest(n=1, check_graph=False)
def _test_clip_flow_with_random_data(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clip(input, min=random().to(float), max=random().to(float))
    return y


@autotest(n=1, check_graph=False)
def _test_clip_min_none_flow_with_random_data(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clip(input, max=random().to(float))
    return y


@autotest(n=1, check_graph=False)
def _test_clip_max_none_flow_with_random_data(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clip(input, min=random().to(float))
    return y


@autotest(n=1, check_graph=False)
def _test_clamp_with_0_size_data(test_case, ndim, zerodim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    dims[zerodim] = 0
    input = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.clamp(input, min=random().to(float), max=random().to(float))
    return y


class TestClampModule(flow.unittest.TestCase):
    @globaltest
    def test_clamp(test_case):
        for placement in all_placement():
            ndim = random(1, 4).to(int).value()
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                _test_clamp_flow_with_random_data(test_case, ndim, placement, sbp)
                _test_clamp_min_none_flow_with_random_data(
                    test_case, ndim, placement, sbp
                )
                _test_clamp_max_none_flow_with_random_data(
                    test_case, ndim, placement, sbp
                )
                _test_clip_flow_with_random_data(test_case, ndim, placement, sbp)
                _test_clip_min_none_flow_with_random_data(
                    test_case, ndim, placement, sbp
                )
                _test_clip_max_none_flow_with_random_data(
                    test_case, ndim, placement, sbp
                )

            zerodim = random(0, ndim).to(int).value()
            valid_split_axis = [i for i in range(ndim) if i != zerodim]
            for sbp in all_sbp(
                placement, max_dim=min(2, ndim), valid_split_axis=valid_split_axis
            ):
                _test_clamp_with_0_size_data(test_case, ndim, zerodim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
