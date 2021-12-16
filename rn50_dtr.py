# coding=utf-8
import numpy as np
from numpy import random
import oneflow as flow
import oneflow.nn as nn

import resnet50_model


# resnet50 bs 32, use_disjoint_set=False: threshold ~800MB

# memory policy:
# 1: only reuse the memory block with exactly the same size
# 2: reuse the memory block with the same size or larger

dtr_enabled = True
threshold = "800MB"
debug_level = 1
memory_policy = 1
use_disjoint_set = True

print(f'dtr_enabled: {dtr_enabled}, threshold: {threshold}, debug_level: {debug_level}, memory_policy: {memory_policy}, use_disjoint_set: {use_disjoint_set}')

flow.enable_dtr(dtr_enabled, threshold, debug_level, memory_policy, use_disjoint_set)

seed = 20
flow.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def sync():
    flow._oneflow_internal.eager.multi_client.Sync()


def display():
    flow._oneflow_internal.dtr.display()


# init model
model = resnet50_model.resnet50(norm_layer=nn.Identity)
# model.load_state_dict(flow.load('/tmp/abcde'))
flow.save(model.state_dict(), '/tmp/abcde')

criterion = nn.CrossEntropyLoss()

cuda0 = flow.device('cuda:0')

# enable module to use cuda
model.to(cuda0)
criterion.to(cuda0)

learning_rate = 1e-3
# optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

batch_size = 32

# generate random data and label
train_data = flow.tensor(
    np.random.uniform(size=(batch_size, 3, 224, 224)).astype(np.float32), device=cuda0
)
train_label = flow.tensor(
    (np.random.uniform(size=(batch_size,)) * 1000).astype(np.int32), dtype=flow.int32, device=cuda0
)

# run forward, backward and update parameters
for epoch in range(300):
    logits = model(train_data)
    loss = criterion(logits, train_label)
    print('forward over')
    # loss.print_ptr()
    loss.backward()
    print('backward over')
    optimizer.step()
    print('step over')
    optimizer.zero_grad(True)
    if debug_level > 0:
        sync()
        display()
    print('loss: ', loss.numpy())

