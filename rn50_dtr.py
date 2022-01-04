# coding=utf-8

import time

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
# full: 1700MB
threshold = "7500MB"
debug_level = 0
memory_policy = 1
heuristic = "eq"

print(f'dtr_enabled: {dtr_enabled}, threshold: {threshold}, debug_level: {debug_level}, memory_policy: {memory_policy}, heuristic: {heuristic}')

flow.enable_dtr(dtr_enabled, threshold, debug_level, memory_policy, heuristic)

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
model.load_state_dict(flow.load('/tmp/abcde'))
# flow.save(model.state_dict(), '/tmp/abcde')

criterion = nn.CrossEntropyLoss()

cuda0 = flow.device('cuda:0')

# enable module to use cuda
model.to(cuda0)
criterion.to(cuda0)

learning_rate = 1e-3
# optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

batch_size = 256

# generate random data and label
train_data = flow.tensor(
    np.random.uniform(size=(batch_size, 3, 224, 224)).astype(np.float32), device=cuda0
)
train_label = flow.tensor(
    (np.random.uniform(size=(batch_size,)) * 1000).astype(np.int32), dtype=flow.int32, device=cuda0
)

# run forward, backward and update parameters
WARMUP_ITERS = 3
ALL_ITERS = 3000
for epoch in range(ALL_ITERS):
    if epoch == WARMUP_ITERS:
        start_time = time.time()
    logits = model(train_data)
    loss = criterion(logits, train_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(True)
    # sync()
    # exit(2)
    if debug_level > 0:
        sync()
        display()
    if (epoch + 1) % 10 == 0:
        print('loss: ', loss.numpy())

end_time = time.time()
print(f'{ALL_ITERS - WARMUP_ITERS} iters: avg {(end_time - start_time) / (ALL_ITERS - WARMUP_ITERS)}s')
