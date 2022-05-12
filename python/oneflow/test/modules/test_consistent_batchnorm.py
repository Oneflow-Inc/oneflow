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


@autotest(n=1, rtol=1e-3, atol=1e-3, check_graph=False)
def _test_batchnorm1d_module(test_case, placement, sbp):
    dims = [random(1, 3).to(int) for _ in range(3)]
    channel = dims[1]
    track_running_stats = random_bool().value()
    m = torch.nn.BatchNorm1d(
        num_features=channel, track_running_stats=track_running_stats
    ).to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))])
    m.train(random())
    x = random_tensor(3, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, rtol=1e-3, atol=1e-3, check_graph=False)
def _test_batchnorm2d_module(test_case, placement, sbp):
    dims = [random(1, 3).to(int) for _ in range(4)]
    channel = dims[1]
    track_running_stats = random_bool().value()
    m = torch.nn.BatchNorm2d(
        num_features=channel, track_running_stats=track_running_stats
    ).to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))])
    m.train(random())
    x = random_tensor(4, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, rtol=1e-3, atol=1e-3, check_graph=False)
def _test_batchnorm3d_module(test_case, placement, sbp):
    dims = [random(1, 3).to(int) for _ in range(5)]
    channel = dims[1]
    track_running_stats = random_bool().value()
    m = torch.nn.BatchNorm3d(
        num_features=channel, track_running_stats=track_running_stats
    ).to_global(placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))])
    m.train(random())
    x = random_tensor(5, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


class TestBatchNormModule(flow.unittest.TestCase):
    @globaltest
    def test_batchnorm(test_case):
        for placement in all_placement():
            # Splitting the input will cause the mean and variance calculated by
            # each rank to be different from PyTorch.
            for sbp in all_sbp(placement, except_split=True):
                _test_batchnorm1d_module(test_case, placement, sbp)
                _test_batchnorm2d_module(test_case, placement, sbp)
                _test_batchnorm3d_module(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
