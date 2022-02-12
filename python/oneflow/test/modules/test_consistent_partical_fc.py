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


class TestParitalFC(flow.unittest.TestCase):
    @consistent
    def test_parital_fc(test_case):
        placement = flow.env.all_device_placement("cuda")
        for sbp in all_sbp(placement, max_dim=2):
            for label_sbp in all_sbp(placement, max_dim=1):

                w =  flow.randn(50000, 128).to_consistent(placement=placement, sbp=sbp)
                label = flow.randint(0, 50000, (512,)).to_consistent(placement=placement, sbp=label_sbp)
                num_sample = 5000

                out = flow.distributed_partial_fc_sample(w, label, num_sample)

                if flow.env.get_rank() == 0:
                    print(out[0].to_local().shape)
                    test_case.assertTrue(out[0].to_local().shape == flow.Size([512]))
                    test_case.assertTrue(out[1].to_local().shape == flow.Size([5000]))
                    test_case.assertTrue(out[2].to_local().shape == flow.Size([5000, 128]))

if __name__ == "__main__":
    unittest.main()

#两卡报错