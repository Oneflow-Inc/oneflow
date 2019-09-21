import oneflow as flow

class RPNHead(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, layers):
        with flow.deprecated.variable_scope("rpn-head"):
            pass