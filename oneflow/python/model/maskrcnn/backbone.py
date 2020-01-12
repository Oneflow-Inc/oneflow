import oneflow as flow
from resnet import ResNet
from fpn import FPN


class Backbone(object):
    def __init__(self, cfg):
        self.resnet = ResNet(cfg)
        self.fpn = FPN(cfg)

    def build(self, in_blob):
        in_blob = flow.nvtx.range_push(in_blob, "backbone")
        with flow.deprecated.variable_scope("backbone"):
            features = self.resnet.build(in_blob)
            layer_features = self.fpn.build(features)
        return layer_features
