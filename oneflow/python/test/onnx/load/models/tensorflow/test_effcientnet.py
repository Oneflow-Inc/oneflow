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
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from oneflow.python.test.onnx.load.util import load_tensorflow2_module_and_check

def test_EfficientNetB0(test_case):
    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()
            self.EfficientNetB0 = EfficientNetB0(weights=None)
        def call(self, x):
            x = self.EfficientNetB0(x)
            return x

    load_tensorflow2_module_and_check(test_case, Net, input_size=(1, 224, 224, 3), train_flag=False)

