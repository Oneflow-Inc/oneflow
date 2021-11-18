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
from random import randint
from random import choice

import numpy as np

from oneflow.test.modules.resnet50_model import resnet50
import oneflow as flow
import oneflow.unittest


res50_module = resnet50(
    replace_stride_with_dilation=[False, False, False], norm_layer=flow.nn.BatchNorm2d,
).eval()


def get_cpu_graph():
    class ResNet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.res50_module = res50_module

        def build(self, x):
            return self.res50_module(x)

    restnet50_g = ResNet50Graph()
    return restnet50_g


def get_cuda_graph():
    class ResNet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.res50_module = res50_module.to(device="cuda")

        def build(self, x):
            return self.res50_module(x)

    restnet50_g = ResNet50Graph()
    return restnet50_g


class TestXrtResNet50(flow.unittest.TestCase):
    def test_xrt_restnet50(test_case):
        x = np.random.random((1, 3, 224, 224)).astype(np.float32)
        x_cpu = flow.tensor(x, dtype=flow.float32, device=flow.device("cpu"))
        x_cuda = flow.tensor(x, dtype=flow.float32, device=flow.device("cuda"))

        resnet50_g = get_cpu_graph()
        out = resnet50_g(x_cpu)

        resnet50_g_openvino = get_cpu_graph()
        resnet50_g_openvino.config.enable_xrt_use_openvino(True)
        out_openvino = resnet50_g_openvino(x_cpu)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_openvino.numpy(), rtol=1e-3, atol=1e-4)
        )

        resnet50_g_tensorrt = get_cuda_graph()
        resnet50_g_tensorrt.config.enable_xrt_use_tensorrt(True)
        out_tensorrt = resnet50_g_tensorrt(x_cuda)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_tensorrt.numpy(), rtol=1e-3, atol=1e-4)
        )

        resnet50_g_xla = get_cuda_graph()
        resnet50_g_xla.config.enable_xrt_use_xla_jit(True)
        out_xla = resnet50_g_xla(x_cuda)
        test_case.assertTrue(
            np.allclose(out.numpy(), out_xla.numpy(), rtol=1e-3, atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
