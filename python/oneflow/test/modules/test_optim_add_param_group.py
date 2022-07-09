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

import tempfile
import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow

def _test_sgd_add_param_group(test_case):
    w1 = flow.ones(3, 3)
    w1.requires_grad = True
    w2 = flow.ones(3, 3)
    w2.requires_grad = True
    o = flow.optim.SGD([w1])
    # print(o.param_groups)
    # {'state': {'step': 0, 0: {}}, 'param_groups': [{'_options': {'lr': 0.001, 'momentum': 0.0, 'dampening': 0.0, 'weight_decay': 0.0, 'nesterov': False, 'maximize': False}, '_enable_clip_grad': False, 'params': [0]}]}
    o.add_param_group({'params': w2})
    # print(o.param_groups)


@flow.unittest.skip_unless_1n4d()
class TestReduce(flow.unittest.TestCase):
    def test_sgd_add_param_group(test_case):
        _test_sgd_add_param_group(test_case)

