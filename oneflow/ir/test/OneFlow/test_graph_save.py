# RUN: python3 %s

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import unittest
import oneflow as flow
import oneflow.unittest

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
    def test_save(self):
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


if __name__ == "__main__":
    unittest.main()
