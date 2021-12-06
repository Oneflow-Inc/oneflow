# RUN: python3 %s

import os
import sys

from google.protobuf import text_format

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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
                model.to_consistent(**placement_arg)
            else:
                model.to(**placement_arg)
        self.model = model

    def build(self, image):
        logits = self.model(image.to("cuda"))
        pred = logits.softmax()
        return pred


@flow.unittest.skip_unless_1n1d()
class GraphSaveTestCase(flow.unittest.TestCase):
    def test_save_and_load(self):
        placement_arg = {
            "placement": flow.placement("cuda", {0: [0]}),
            "sbp": flow.sbp.broadcast,
        }
        graph = InferGraph(placement_arg)
        image_placeholder = flow.empty(
            (1, 3, 224, 224),
            dtype=flow.float32,
            placement=flow.placement("cpu", {0: [0]}),
            sbp=flow.sbp.broadcast,
        )
        graph._compile(image_placeholder)
        graph.save("saved_model")

        saved_path = os.path.join("saved_model", graph.name + ".mlir")
        serialized_job = oneflow._oneflow_internal.nn.graph.LoadSerializedJobFromIR(
            saved_path
        )
        job = job_pb.Job()
        job.ParseFromString(serialized_job)

        op_list = []
        op_list_ = []

        for op in job.net.op:
            op_list.append(op)

        for op in graph._forward_job_proto.net.op:
            op_list_.append(op)

        def sort_by_op_name(op):
            return op.name

        op_list.sort(key=sort_by_op_name)
        op_list_.sort(key=sort_by_op_name)

        for (op, op_) in zip(op_list, op_list_):
            self.assertTrue(op == op_)
            # if op != op_:
            #     print(op)
            #     print("-" * 20)
            #     print(op_)
            #     self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
