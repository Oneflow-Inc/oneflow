import oneflow as flow
import oneflow.nn as nn
import numpy as np
import copy
import json
from resnet50 import Bottleneck , resnet50


np.random.seed(1)
inputs_np = np.random.randn(4, 64, 56, 56)

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
    if load:
        load_weight(res50)
    models = res50.layer1
    return models

model = get_module()
cpu_dict = {}
inputs_cpu = flow.tensor(inputs_np, dtype=flow.float32, requires_grad=True)
state_dict = copy.deepcopy(model.state_dict())
out = model(inputs_cpu)
# out = res50.layer1[1](out)
# print(out)
# out_np = out.numpy()
# np.save("layer1-0", out_np)
loss = out.sum()
loss.backward()
print(inputs_cpu.grad)
print("split------------------")
for name, param in model.named_parameters():
    cpu_dict["[grad]:" + name] = param.grad


npu_dict = {}
inputs_npu_g = flow.tensor(inputs_np, dtype=flow.float32, requires_grad=True)
inputs_npu = inputs_npu_g.to("npu")
model_npu = get_module(False)
model_npu.load_state_dict(state_dict)
model_npu = model_npu.to("npu")
model_npu.zero_grad()
out = model_npu(inputs_npu)
print(out)
print("split------------------")
loss = out.sum()
loss.backward()
print(inputs_npu_g.grad)
print("split------------------")
for name, param in model_npu.named_parameters():
    npu_dict["[grad]:" + name] = param.grad


def compare(x1, x2, prefix=''):
    if isinstance(x1, tuple):
        if x1:
            for idx in range(len(x1)):
                try:
                    compare(x1[idx], x2[idx], prefix=prefix + '.%d' % idx)
                except Exception as e:
                    # print(str(e))
                    print(prefix, 'failed.')
    elif isinstance(x1, flow.Tensor) and isinstance(x2, flow.Tensor):
        try:
            l1_error = (x1.float() - x2.to("cpu")).abs().mean()
            rel_error = l1_error / (x1.abs().mean())
            print(prefix, 'l1_error: ', l1_error, 'rel_error', rel_error)
            if l1_error * rel_error > 10 :
                print('\n###\n',prefix, 'should checked!','\n###\n')
        except Exception as e:
            print(str(e))
            print(prefix, 'failed.')

for k in cpu_dict:
    compare(cpu_dict[k], npu_dict[k], prefix=k)