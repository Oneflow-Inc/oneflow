from resnet50 import resnet50
import oneflow as flow
import numpy as np

use_cpu = 0

cross_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

inputs_g = flow.randn(32,10,requires_grad=True)
np_labels = np.ones(32,dtype=np.int32)
labels = flow.tensor(np_labels)

if use_cpu:
    out = cross_entropy(inputs_g,labels)
    out.backward()
    print(inputs_g.grad)
else:
    labels = labels.to("npu")
    inputs = inputs_g.to("npu").to(flow.float16)
    out = cross_entropy(inputs, labels)
    print(out)
    out.backward()
    print(inputs_g.grad)
