import oneflow.mock_torch as mock

mock.enable()

import torch
from torch import nn

@autotest()
def testcase4module():
    model = nn.Sequential(
    nn.Linear(5, 3),
    nn.Linear(3, 1)
    )
    if isinstance(model, torch.jit.ScriptModule):
        print(True)
    else:
        print(False)
#原先报错为ModuleNotFoundError: No module named 'oneflow.jit.__ScriptModule'
#修改后打印信息为false，修改完毕

