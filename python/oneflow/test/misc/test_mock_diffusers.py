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

"""
If some modules import torch internally, 
flow.mock_torch.disable() should be able to restore the original torch within these modules.
"""


class TestMock(flow.unittest.TestCase):
    def test_mock_diffusers(test_case):

        flow.mock_torch.enable(lazy=True)
        from diffusers import UNet2DConditionModel

        torch_module = UNet2DConditionModel.__dict__["forward"].__globals__["torch"]

        flow.mock_torch.disable()
        from diffusers import UNet2DConditionModel

        torch_module = UNet2DConditionModel.__dict__["forward"].__globals__["torch"]

        # check whether the torch module is the original torch
        test_case.assertFalse(isinstance(torch_module, flow.mock_torch.ModuleWrapper))


if __name__ == "__main__":
    unittest.main()
