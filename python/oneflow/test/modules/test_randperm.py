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
import oneflow as flow
import oneflow.unittest
from collections import OrderedDict
from test_util import GenArgList
import numpy as np
from oneflow.nn.module import Module
import oneflow.nn as nn
def _test_randperm_with_generator(test_case, N, device):
    generator = flow.Generator()
    generator.manual_seed(0)
    y_1 = flow.randperm(N, device=device, generator=generator)
    generator.manual_seed(0)
    y_2 = flow.randperm(N, device=device, generator=generator)
    test_case.assertTrue(np.allclose(y_1.numpy(), y_2.numpy()))

def _test_randperm_backward(test_case, N, device):
    x = flow.randperm(N, device=device)
    x.requires_grad = True
    y = x.sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.ones(N), 1e-05, 1e-05))

@flow.unittest.skip_unless_1n1d()
class Testrandperm(flow.unittest.TestCase):
    def test_randperm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_functions"] = [
            _test_randperm_with_generator,
            _test_randperm_backward
        ]
        arg_dict["N"] = [i for i in range(2, 2, 10)]
        arg_dict["device"] = ["cpu","gpu"]
        # @TODO:GPU version test need context support from backend
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
