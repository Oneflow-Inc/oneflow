import oneflow.nn as nn
import oneflow as flow
import numpy as np


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
class Bottleneck1(nn.Module):
    def __init__(self):
        super(Bottleneck1, self).__init__()
        self.conv1 = conv1x1(64, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2  = conv3x3(64, 64, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv1x1(64,256)
        self.bn3 = nn.BatchNorm2d(256)
        self.downsampleConv = conv1x1(64,256)
        self.downsampleBN = nn.BatchNorm2d(256)
    def forward(self, x):
        identity = self.downsampleConv(x)
        identity = self.downsampleBN(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        return out
        
class Bottleneck2(nn.Module):
    def __init__(self):
        super(Bottleneck2, self).__init__()
        self.conv1 = conv1x1(256, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2  = conv3x3(64, 64, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv1x1(64,256)
        self.bn3 = nn.BatchNorm2d(256)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out) 
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + x
        return out        

class BN(nn.Module):
    def __init__(self):
        super(BN, self).__init__()
        self.bt1 = Bottleneck1()
        self.bt2 = Bottleneck2()
    def forward(self, x):
        out = self.bt1(x)
        out = self.bt2(out)
        return out
np.random.seed(1)
inputs_np = np.random.randn(4, 64, 56, 56)
npu_dict = {}
inputs_npu_g = flow.tensor(inputs_np, dtype=flow.float32, requires_grad=True)
inputs_npu = inputs_npu_g.to("npu")
model = BN().to("npu")
# for name, param in model.named_parameters():
#     if name=='bt2.bn1.weight':
#         print("step in")
#         param.grad = flow.zeros(param.shape).to("npu")
out = model(inputs_npu)

loss = out.sum()
loss.backward()

for name, param in model.named_parameters():
    npu_dict["[grad]:" + name] = param.grad
def compare(x1, prefix=''):
    if isinstance(x1, tuple):
        if x1:
            for idx in range(len(x1)):
                try:
                    compare(x1[idx], prefix=prefix + '.%d' % idx)
                except Exception as e:
                    # print(str(e))
                    print(prefix, 'failed.')
    elif isinstance(x1, flow.Tensor):
        try:
            #l1_sum = x1.to("cpu").abs().sum()
            # if l1_sum < 1 :
            #     print('\n###\n',prefix, 'should checked!','\n###\n')
            #     print(x1)
            if prefix.find('bn')!=-1:
                print(prefix)
                print(x1)
        except Exception as e:
            print(str(e))
            print(prefix, 'failed.')

for k in npu_dict:
    compare(npu_dict[k], prefix=k)
