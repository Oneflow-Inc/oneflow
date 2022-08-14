from resnet50 import resnet50
import oneflow as torch
import json
import numpy as np
np.random.seed(1)
model = resnet50(num_classes=10)
cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
def gen_weight():
    ms = model.state_dict()
    for k in ms:
        ms[k] = ms[k].numpy().tolist()
    f = open("torch_resnet50.json", "wb")
    f.write(json.dumps(ms).encode())
    f.close()
params=model.state_dict() 
# for k,v in params.items():
#     print(k) 
# print("----------------------------")
def load_weight():
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
        ckpt[key] = torch.from_numpy(w)
    model.load_state_dict(ckpt,strict=True)
inp = np.random.randn(4,3,224,224)
load_weight()
inputs = torch.tensor(inp,dtype=torch.float32,requires_grad=True)
inputs_n = inputs.to("npu")
model = model.to("npu")
optimizer = torch.optim.TORCH_SGD(model.parameters(), lr = 0.01, momentum=0.9)
labels = torch.ones(4,dtype=torch.int32).to('npu')
out = model(inputs_n)
loss = cross_entropy(out, labels)
loss.backward()
optimizer.step()




