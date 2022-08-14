import oneflow as flow
import oneflow.nn as nn
import numpy as np
import copy
import json
import time
from resnet50 import Bottleneck , resnet50
from apex_oneflow import initialize

def get_module(load=True):
    res50 = resnet50(num_classes = 10)
    res50 = res50.train()
    models = res50

    return models

model_npu = get_module()
model_npu = model_npu.to("npu")
model_npu = initialize(model_npu)

#start = time.time()
for i in range(10):
    
    inputs_npu = flow.randn(4, 3, 224, 224).to('npu').to(flow.float16)
    out = model_npu(inputs_npu)
    loss = out.sum()
    loss.backward()
    model_npu.zero_grad()
#print(time.time()-start)