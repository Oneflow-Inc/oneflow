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

def test_concat(test_case):
    class Net(tf.keras.Model):
        def call(self, x):
            y = x * 3
            return tf.keras.layers.Concatenate()([x, y])

    load_tensorflow2_module_and_check(test_case, Net)


def test_concat_with_axis(test_case):
    class Net(tf.keras.Model):
        def call(self, x):
            y = x * 3
            return tf.keras.layers.Concatenate(axis=1)([x, y])

    load_tensorflow2_module_and_check(test_case, Net)


def test_unsqueeze(test_case):
    class Net(tf.keras.Model):
        def call(self, x):
            return tf.expand_dims(x, axis=2)

    load_tensorflow2_module_and_check(test_case, Net)


def test_transpose(test_case):
    class Net(tf.keras.Model):
        def call(self, x):
            # shape = x.shape
            return tf.transpose(x, perm=[0, 3, 1, 2])

    load_tensorflow2_module_and_check(test_case, Net)


def test_gather(test_case):
    class Net(tf.keras.Model):
        def call(self, x):
            return x[1]

    load_tensorflow2_module_and_check(test_case, Net)


def test_tensor_index(test_case):
    class Net(tf.keras.Model):
        def call(self, x):
            return x[0, 1:3, :1, 2:4]

    load_tensorflow2_module_and_check(test_case, Net)

from absl import app
from absl.testing import absltest

test_case = absltest.TestCase
test_concat(test_case)