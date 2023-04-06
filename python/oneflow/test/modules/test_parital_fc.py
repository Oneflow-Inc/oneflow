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

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


# TODO: guoran, fix this on multi gpu
@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestParitalFC(flow.unittest.TestCase):
    def test_parital_fc(test_case):
        p = flow.placement.all("cuda")
        w = flow.randn(
            50000, 128, placement=p, sbp=flow.sbp.broadcast, requires_grad=True
        )
        label = flow.randint(0, 50000, (512,), placement=p, sbp=flow.sbp.broadcast)
        num_sample = 5000
        out = flow.distributed_partial_fc_sample(w, label, num_sample)
        test_case.assertTrue(out[0].shape == flow.Size([512]))
        test_case.assertTrue(out[1].shape == flow.Size([5000]))
        test_case.assertTrue(out[2].shape == flow.Size([5000, 128]))
        # test gradient function
        sample_weight = out[2]
        sample_weight.sum().backward()


if __name__ == "__main__":
    unittest.main()
