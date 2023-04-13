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


@autotest(n=1, auto_backward=False, check_graph=True)
def do_test_eye_impl(test_case, placement, sbp):
    n = random(1, 5).to(int).value() * 8
    m = random(1, 5).to(int).value() * 8
    x = torch.eye(n, m)
    x.oneflow = flow.tensor(
        x.pytorch.cpu().detach().numpy(),
        requires_grad=x.pytorch.requires_grad,
        placement=placement,
        sbp=sbp,
    )
    return x


class TestEyeGlobal(flow.unittest.TestCase):
    @globaltest
    def test_eye(test_case):
        shape = random_tensor().shape
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                do_test_eye_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
