"""用于精度比对
"""

import oneflow as flow
import oneflow.nn as nn
from resnet50 import resnet50
import copy
from apex_oneflow import initialize
def get_model():
    model = resnet50(num_classes=1000)
    return model

input_tensor = flow.randn(2, 3, 224, 224,requires_grad=True)

# 设置npu_device
npu_device = 'npu'

def cri_func(x):
    base_func = nn.CrossEntropyLoss()
    T = flow.tensor([10,89]).to(x.device)
    if str(T.device).startswith('npu'):
        T = T.int()
    return base_func(x, T)

# 设置hook
def hook_func(name, save_dict, module):
    def hook_function(module, inputs, outputs):
        inputs_key = name + '_inputs'
        idx = 0
        while inputs_key in save_dict:
            inputs_key = inputs_key.split('-')[0] + '-%d'%idx
            idx +=1
        save_dict[inputs_key] = inputs

        outputs_key = name + '_outputs'
        idx = 0
        while outputs_key in save_dict:
            outputs_key = outputs_key.split('-')[0] + '-%d'%idx
            idx +=1
        save_dict[outputs_key] = outputs
    return hook_function

model = get_model()
#optimizer = torch.optim.SGD(model.parameters(), 0.1)
state_dict = copy.deepcopy(model.state_dict())

# CPU注册hook，cpu_dict用于存储对比对象
cpu_dict = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]:' + name, cpu_dict, module))

# CPU运行正反向，获取正反向每个module的输入输出和所有参数的grad
out = model(input_tensor)
loss = cri_func(out)
loss.backward()
for name, param in model.named_parameters():
    cpu_dict["[grad]:" + name] = param.grad

model = get_model()
model.load_state_dict(state_dict)

npu_dict = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]:' + name, npu_dict, module))

model = model.to("npu")
model = initialize(model)
input_tensor = input_tensor.to("npu").to(flow.float16)

out = model(input_tensor)
loss = cri_func(out)

loss.backward()
for name, param in model.named_parameters():
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
                #print(x1)
                #print("----------------------------")
                #print(x2)
                print('\n###\n',prefix, 'should checked!','\n###\n')
        except Exception as e:
            print(str(e))
            print(prefix, 'failed.')

for k in cpu_dict:
    compare(cpu_dict[k], npu_dict[k], prefix=k)


