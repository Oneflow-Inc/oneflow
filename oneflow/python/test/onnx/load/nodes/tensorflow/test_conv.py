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
import tensorflow as tf

from oneflow.python.test.onnx.load.util import load_tensorflow2_module_and_check


def test_conv2d_k3s1p1(test_case):
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = tf.keras.layers.Conv2D(5, 3, padding="same")

        def call(self, x):
            x = self.conv(x)
            return x

    load_tensorflow2_module_and_check(test_case, Net, input_size=(2, 4, 3, 5))


def test_conv2d_k3s1p0(test_case):
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = tf.keras.layers.Conv2D(5, 3, padding="valid")

        def call(self, x):
            x = self.conv(x)
            return x

    load_tensorflow2_module_and_check(test_case, Net, input_size=(2, 4, 3, 5))


def test_conv2d_k3s2p0(test_case):
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = tf.keras.layers.Conv2D(5, 3, strides=(2, 2), padding="valid")

        def call(self, x):
            x = self.conv(x)
            return x

    load_tensorflow2_module_and_check(test_case, Net, input_size=(2, 4, 9, 7))


# def test_conv2d_k3s2p0g2(test_case):
#     class Net(tf.keras.Model):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv = tf.keras.layers.Conv2D(1, 3, strides=(1, 1), padding="valid", groups=6)

#         def call(self, x):
#             x = self.conv(x)
#             return x

#     load_tensorflow2_module_and_check(test_case, Net, input_size=(2, 4, 9, 6))


# def test_conv2d_k3s2p0g2d2(test_case):
#     class Net(tf.keras.Model):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.conv = tf.keras.layers.Conv2D(6, 3, strides=(1, 1), padding="valid", groups=2, dilation_rate=2)

#         def call(self, x):
#             x = self.conv(x)
#             return x

#     load_tensorflow2_module_and_check(test_case, Net, input_size=(2, 4, 13, 12))

