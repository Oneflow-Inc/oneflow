import oneflow as flow
import oneflow.nn as nn
import time

flow.manual_seed(1)

use_npu = 1


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Conv2d(3,16,(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(802816,10,bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
# if use_cpu:
#     model = TestNet()
#     inputs = flow.randn(1,3,224,224,requires_grad=True,dtype=flow.float32)
#     print(inputs)
#     out = model(inputs)
#     #print(out)
# else:
#     model = TestNet().to("npu")
#     inputs = flow.randn(1,3,224,224,requires_grad=True,dtype=flow.float32).to("npu")
#     model(inputs)

model = TestNet()
inputs = flow.randn(1,3,224,224,requires_grad=True,dtype=flow.float32)
out = model(inputs)
print(out)
print("---------------------------------------------")
if use_npu:
    model = model.to("npu")
    inputs = inputs.to("npu")
    out = model(inputs)