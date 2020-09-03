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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import oneflow as flow
import torch
from torch import nn
import torch.nn.functional as F
import io
import tempfile
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(3, padding=1)
        self.linear = nn.Linear(8, 4, True)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.register_buffer("label", torch.tensor([0, 1], dtype=torch.int64))
        self.label = torch.tensor([3, 1], device=torch.device("cuda"))

    def forward(self, x):
        # x = self.bn(self.conv(x))
        # x = self.pool(x)
        x = torch.flatten(x, 1)
        # x = self.linear(x)
        # x = self.loss(x, self.label)
        return x


model = Net()

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.train.primary_lr(0)
func_config.train.model_update_conf(dict(naive_conf={}))
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
input_size = (2, 4, 5, 3)
# input_size = (2, 2)

# job = torch2flow(model, func_config, input_size)
@flow.global_function(func_config)
def job(x=flow.FixedTensorDef(input_size)):
    x += flow.get_variable(
        name="trick",
        shape=(1,),
        dtype=flow.float,
        initializer=flow.zeros_initializer(),
    )

    y = flow.from_pytorch(model, x)
    flow.losses.add_loss(y)
    return y


checkpoint = flow.train.CheckPoint()
checkpoint.load("/tmp/tmp2")

model = model.to("cuda")
ipt1 = np.random.uniform(low=-1000, high=1000, size=input_size).astype(np.float32)
# flow_res = temp_job(ipt1).get().ndarray()
flow_res = job(ipt1).get().ndarray()
pytorch_res = model(torch.tensor(ipt1).to("cuda")).cpu().detach().numpy()
print("ipt:")
print(ipt1)
print("flow res:")
print(flow_res)
print("pytorch res:")
print(pytorch_res)
a, b = flow_res.flatten(), pytorch_res.flatten()

max_idx = np.argmax(np.abs(a - b) / a)
print("max rel diff is {} at index {}".format(np.max(np.abs(a - b) / a), max_idx))
print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
assert np.allclose(flow_res, pytorch_res, rtol=1e-4, atol=1e-5)
