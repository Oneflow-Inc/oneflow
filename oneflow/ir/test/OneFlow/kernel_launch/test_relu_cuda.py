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
# RUN: python3 %s | FileCheck %s
# CHECK: oneflow.kernel_launch
import numpy as np
import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
import oneflow as flow


class GraphToRun(flow.nn.Graph):
    def __init__(self):
        super().__init__()

    def build(self, x):
        return flow.relu(x)


x = flow.Tensor([1, -1]).cuda()
graph_to_run = GraphToRun()
lazy_relu = graph_to_run(x)


def assert_equal(expected, got):
    assert flow.all(flow.equal(expected, got)), {"expected": expected, "got": got}


assert_equal(flow.Tensor([1, 0]).cuda(), lazy_relu)
