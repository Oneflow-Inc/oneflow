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

import numpy as np
import torch
import torchvision.models as models_torch
import flowvision.models as models_flow

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestResNet18LoadWeightCompatibile(flow.unittest.TestCase):
    def test_resnet18_load_weight_compatibile(test_case):
        resnet18_torch = models_torch.resnet18(pretrained=True)
        resnet18_flow = models_flow.resnet18()
        parameters = resnet18_torch.state_dict()
        for key, value in parameters.items():
            val = value.detach().cpu().numpy()
            parameters[key] = val

        resnet18_flow.load_state_dict(parameters)
        torch_input = torch.randn(1, 3, 224, 224)
        flow_input = flow.tensor(torch_input.cpu().numpy())
        torch_output = resnet18_torch(torch_input)
        flow_output = resnet18_flow(flow_input)
        test_case.assertTrue(
            np.allclose(
                torch_output.detach().numpy(), flow_output.numpy(), atol=1e-4, rtol=1e-3
            )
        )


if __name__ == "__main__":
    unittest.main()
