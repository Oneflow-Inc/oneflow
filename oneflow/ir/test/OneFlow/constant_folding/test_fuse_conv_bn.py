import os
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn
from flowvision.models.resnet import resnet50

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"


def _test_fuse_conv_bn(test_case):
    data = flow.randn(1, 3, 224, 224)

    model = resnet50(pretrained=True, progress=True)
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
        np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-5, atol=1e-5)
    )


@flow.unittest.skip_unless_1n1d()
class TestFuseConvBn(oneflow.unittest.TestCase):
    def test_fuse_conv_bn(test_case):
        _test_fuse_conv_bn(test_case)


if __name__ == "__main__":
    unittest.main()
