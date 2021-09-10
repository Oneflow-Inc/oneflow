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
import tempfile
import oneflow as flow
import oneflow.nn as nn
import unittest
import numpy as np
from oneflow.fx import symbolic_trace
from oneflow.fx.passes.quantization import QuantizationAwareTraining

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_inplace(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features[0](x)
        x = flow.neg(x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = flow.neg(x)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = flow.neg(x)
        x = self.features[7](x)
        x = self.features[8](x)
        x = flow.neg(x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = flow.neg(x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x

@flow.unittest.skip_unless_1n1d()
class TestAlexNet(flow.unittest.TestCase):
    def test_alexnet(test_case):
        m = AlexNet()
        gm: flow.fx.GraphModule = symbolic_trace(m)
        input = flow.randn(1, 3, 224, 224)
        insert_place = QuantizationAwareTraining(gm).propagate(input)
        for x in gm.graph.nodes:
            if x.target in insert_place:
                y = x._next
                with gm.graph.inserting_after(x):
                    neg : flow.fx.Node = gm.graph.call_function(the_function=flow.neg, args=(x, ))
                    _, *nxt_args = y.args
                    y.args = (neg, *nxt_args)

        gm.recompile()
        assert np.allclose(gm(input).numpy(), m.forward_inplace(input).numpy(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
