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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


def test_conv2d_k3s1p1(test_case):
    class Net(nn.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(4, 5, 3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    load_paddle_module_and_check(test_case, Net, input_size=(2, 4, 3, 5))


def test_conv2d_k3s1p0(test_case):
    class Net(nn.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(4, 5, 3, padding=0)

        def forward(self, x):
            x = self.conv(x)
            return x

    load_paddle_module_and_check(test_case, Net, input_size=(2, 4, 3, 5))


def test_conv2d_k3s2p0(test_case):
    class Net(nn.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(4, 5, 3, stride=2, padding=0)

        def forward(self, x):
            x = self.conv(x)
            return x

    load_paddle_module_and_check(test_case, Net, input_size=(2, 4, 9, 7))


def test_conv2d_k3s2p0g2(test_case):
    class Net(nn.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(4, 6, 3, stride=1, padding=1, groups=2)

        def forward(self, x):
            x = self.conv(x)
            return x

    load_paddle_module_and_check(test_case, Net, input_size=(2, 4, 9, 7))


def test_conv2d_k3s2p0g2d2(test_case):
    class Net(nn.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(4, 6, 3, stride=1, padding=1, groups=2, dilation=2)

        def forward(self, x):
            x = self.conv(x)
            return x

    load_paddle_module_and_check(test_case, Net, input_size=(2, 4, 13, 12))
