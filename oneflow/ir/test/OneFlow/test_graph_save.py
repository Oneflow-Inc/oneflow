import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import unittest
import oneflow as flow

from networks.resnet50 import resnet50


class InferGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = resnet50()

    def build(self, image):
        logits = self.model(image.to("cuda"))
        pred = logits.softmax()
        return pred


@flow.unittest.skip_unless_1n1d()
class GraphSaveTestCase(flow.unittest.TestCase):
    def test_save(self):
        graph = InferGraph()
        image_placeholder = flow.empty(
            (1, 3, 224, 224),
            dtype=flow.float32,
            placement=flow.placement("cuda", {0: [0]}),
            sbp=flow.sbp.broadcast,
        )
        graph._compile(image_placeholder)
        graph.save("saved_model")


if __name__ == "__main__":
    unittest.main()
