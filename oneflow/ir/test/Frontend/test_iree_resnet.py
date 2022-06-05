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
# RUN: python3 %s

from oneflow_iree.compiler import Runner
from flowvision.models import resnet50
import oneflow as flow
import oneflow.unittest
import unittest
import os
import numpy as np

from google.protobuf import text_format


os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

model = resnet50(pretrained=True)
model.eval()


class GraphModule(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model

    def build(self, x):
        return self.model(x)


func = Runner(GraphModule, return_numpy=True)
data = np.ones([1, 3, 224, 224]).astype(np.float32)
input = flow.tensor(data, requires_grad=False)
output = func(input)
print(output)
