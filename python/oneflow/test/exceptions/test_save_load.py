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
import tempfile

import oneflow as flow
import oneflow.unittest
import torch


class TestSaveLoad(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_support_pytorch_with_global_src_rank(test_case):
        conv_torch = torch.nn.Conv2d(3, 3, 3)
        conv_flow = flow.nn.Conv2d(3, 3, 3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(conv_torch.state_dict(), f.name)
            with test_case.assertRaises(NotImplementedError) as ctx:
                conv_flow.load_state_dict(flow.load(f.name, support_pytorch=False))
        test_case.assertTrue("No valid load method found" in str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
