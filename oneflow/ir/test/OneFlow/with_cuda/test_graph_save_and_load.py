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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/..")

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"

import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.core.job import job_pb2 as job_pb

from networks.resnet50 import resnet50


class InferGraph(flow.nn.Graph):
    def __init__(self, placement_arg=None):
        super().__init__()
        model = resnet50()
        if placement_arg is not None:
            if "placement" in placement_arg:
                model.to_global(**placement_arg)
            else:
                model.to(**placement_arg)
        self.model = model

    def build(self, image):
        logits = self.model(image.to("cuda"))
        pred = logits.softmax()
        return pred


@unittest.skipIf(not flow.sysconfig.with_mlir(), "only test with mlir")
@flow.unittest.skip_unless_1n1d()
class GraphSaveTestCase(flow.unittest.TestCase):
    def test_save_and_load(self):
        placement_arg = {
            "placement": flow.placement("cuda", ranks=[0]),
            "sbp": flow.sbp.broadcast,
        }
        graph = InferGraph(placement_arg)
        image_placeholder = flow.empty(
            (1, 3, 224, 224),
            dtype=flow.float32,
            placement=flow.placement("cpu", ranks=[0]),
            sbp=flow.sbp.broadcast,
        )
        graph._compile(image_placeholder)
        saved_path = os.path.join("saved_model", graph.name)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        flow.save(graph, saved_path)

        saved_ir_path = os.path.join(saved_path, "model.mlir")
        serialized_job = oneflow._oneflow_internal.nn.graph.LoadSerializedJobFromIR(
            saved_ir_path
        )
        job = job_pb.Job()
        job.ParseFromString(serialized_job)

        # TODO: run loaded job as graph and original graph, compare the result


if __name__ == "__main__":
    unittest.main()
