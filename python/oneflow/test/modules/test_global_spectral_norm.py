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

from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@autotest(check_graph=False)
def _test_global_spectral_norm_with_random_data(test_case, placement, sbp):
    input = random(1, 10).to(int)
    output = random(1, 10).to(int)
    m = torch.nn.Linear(input, output).to_global(placement, sbp)
    m = torch.nn.utils.spectral_norm(m)
    return m.weight_orig


class TestGlobalSpectralNorm(flow.unittest.TestCase):
    @globaltest
    def test_global_spectral_norm_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement):
                _test_global_spectral_norm_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
