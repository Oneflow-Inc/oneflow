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

import numpy as np
from oneflow.test_utils.automated_test_util.generators import constant, nothing
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from automated_test_util import *

@flow.unittest.skip_unless_1n1d()
class TestRNNModule(flow.unittest.TestCase):
    @autotest(n=200)
    def test_rnn_with_random_data(test_case):
        m = torch.nn.RNN(
            input_size = constant(10),
            hidden_size =constant(20),
            num_layers = random(low=1, high=3) | nothing(),
            nonlinearity = constant("relu") | nothing(),
            bias = random_bool() | nothing(),
            batch_first = random_bool() | nothing(),
            dropout = random() | nothing(),
            bidirectional = random_bool() | nothing()
        )
        m.train(random())
        device = random_device()
        print(m)
        m.to(device)
        input = random_pytorch_tensor(ndim=3, dim0=5, dim1=3, dim2=10).to(device)
        h0 = random_pytorch_tensor(ndim=3, dim0=2, dim1=3, dim2=20).to(device)
        output, hn = m(input, h0)
        return output

if __name__ == "__main__":
    unittest.main()
