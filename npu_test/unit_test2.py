import oneflow as flow
import oneflow.nn as nn
import numpy as np
import copy
import json
import time
from resnet50 import Bottleneck , resnet50


np.random.seed(1)
inputs_np = np.random.randn(4, 64, 56, 56)
#inputs_np = np.random.randn(4, 256, 56, 56)
#inputs_np = np.load('layer1-0.npy')

#inputs_np = np.random.randn(4, 3, 224, 224)
def load_weight(m):
    with open('/data/torch_test/torch_resnet50.json','r') as f:
        lines = f.readlines()
        ckpt = json.loads(lines[0])
    changed_key = []
    for key in ckpt.keys():
        if key.find('num_batches_tracked')!=-1:
            changed_key.append(key)
    for key in changed_key:
        ckpt.pop(key)

    for key in ckpt.keys():
        w = ckpt[key]
        w = np.array(w,dtype=np.float32)
        ckpt[key] = flow.from_numpy(w)
    m.load_state_dict(ckpt,strict=False)


def get_module(load=True):
    res50 = resnet50(num_classes = 10)
    res50 = res50.train()
    if load:
        load_weight(res50)
    models = res50.layer1

    return models

npu_dict = {}
inputs_npu_g = flow.tensor(inputs_np, dtype=flow.float32, requires_grad=True)

inputs_npu = inputs_npu_g#.to("npu")
model_npu = get_module()

model_npu = model_npu#.to("npu")
out = model_npu(inputs_npu)
#print(out)
loss = out.sum()
for name, param in model_npu.named_parameters():
    param.grad = flow.zeros(param.shape).to("npu")
loss.backward()
for name, param in model_npu.named_parameters():
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
            # l1_sum = x1.to("cpu").abs().sum()
            # if l1_sum < 1 :
            #     print('\n###\n',prefix, 'should checked!','\n###\n')
            #     print(x1)
            if prefix.find('bn')!=-1:
                print(prefix)
                print(x1)
                #flow._C.identity(x1)
        except Exception as e:
            print(str(e))
            print(prefix, 'failed.')

for k in npu_dict:
    compare(npu_dict[k], prefix=k)