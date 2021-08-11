import os
import argparse

import oneflow as flow
import oneflow.nn as nn
import oneflow.F as F
import oneflow.utils as utils
import oneflow.utils.vision as vision
import oneflow.optim as optim
import oneflow.distributed as dist
from oneflow.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    # ref:https://zhuanlan.zhihu.com/p/178402798
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_dataset():
    transform = vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "cifar10"
    )
    my_trainset = vision.datasets.CIFAR10(root=data_dir, train=True, 
        download=True, transform=transform)

    train_sampler = utils.data.distributed.DistributedSampler(my_trainset)
    trainloader = utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=0, sampler=train_sampler)
    return trainloader


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
if dist.get_rank() == 0:
    local_rank = flow.device("cuda:0")
else:
    local_rank = flow.device("cuda:1")



trainloader = get_dataset()

model = ToyModel().to(local_rank)
# DDP model
model = DDP(model)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss().to(local_rank)


model.train()
for epoch in range(1):
    trainloader.sampler.set_epoch(epoch)
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()

    print("loss = %0.3f" % loss.numpy())

################
# export CUDA_VISIBLE_DEVICES="0,1"
# python -m oneflow.distributed.launch --nproc_per_node 2 test_ddp_flow.py
