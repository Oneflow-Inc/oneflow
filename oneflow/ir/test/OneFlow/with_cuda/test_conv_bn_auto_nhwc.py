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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK: oneflow.transpose

import os
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn
from flowvision.models.resnet import resnet50

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_NORMALIZATION_OPS"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"


def _test_fuse_conv_bn(test_case, with_cuda):
    data = flow.randn(1, 3, 224, 224)
    if with_cuda:
        data = data.to("cuda")

    model = resnet50(pretrained=False, progress=True)
    if with_cuda:
        model.to("cuda")
    model.eval()
    eager_res = model(data)

    class Resnet50Graph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, *input):
            return self.model(*input)

    graph = Resnet50Graph()
    lazy_res = graph(data)

    test_case.assertTrue(
        np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-2, atol=1e-2)
    )


@flow.unittest.skip_unless_1n1d()
class TestFuseConvBn(oneflow.unittest.TestCase):
    @unittest.skipUnless(oneflow.sysconfig.with_cuda(), "only test cpu cases")
    def test_fuse_conv_bn_cuda(test_case):
        _test_fuse_conv_bn(test_case, True)


if __name__ == "__main__":
    unittest.main()
