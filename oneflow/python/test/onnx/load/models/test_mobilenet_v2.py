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
import torchvision

from oneflow.python.test.onnx.load.util import load_pytorch_module_and_check

def test_mobilenet_v2(test_case):
    load_pytorch_module_and_check(
        test_case,
        torchvision.models.mobilenet_v2,
        input_size=(1, 3, 224, 224),
        input_min_val=0,
        input_max_val=1,
        train_flag=False,
    )
