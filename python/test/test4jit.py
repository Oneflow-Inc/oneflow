import oneflow.mock_torch as mock

mock.enable()

import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(5, 3),
    nn.Linear(3, 1)
)

if isinstance(model, torch.jit.ScriptModule):
    print(True)
else:
    print(False)
#原先报错为ModuleNotFoundError: No module named 'oneflow.jit.__ScriptModule'
#修改后打印信息为TypeError: isinstance() arg 2 must be a type or tuple of types，说明torch.jit.ScriptModule已成功返回空对象，空接口撰写完毕

