# coding=utf-8

import time
import os

import numpy as np
from numpy import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# resnet50 bs 32, use_disjoint_set=False: threshold ~800MB

# memory policy:
# 1: only reuse the memory block with exactly the same size
# 2: reuse the memory block with the same size or larger

# os.environ["OF_DTR"] = "1"
# os.environ["OF_DTR_THRESHOLD"] = "3500mb"
# os.environ["OF_DTR_DEBUG"] = "0"
# os.environ["OF_DTR_LR"] = "1"
# os.environ["OF_DTR_BS"] = "80"
# os.environ["OF_ITERS"] = "40"

import argparse


class DataLogger(object):
    """Average data logger."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


class NegativeArgAction(argparse.Action):
    def __init__(self, option_strings, dest, env_var_name, **kwargs):
        assert len(option_strings) == 1
        assert '--no-' in option_strings[0]
        dest = dest[3:]
        super(NegativeArgAction, self).__init__(option_strings, dest, nargs=0, default=True, **kwargs)
        self.env_var_name = env_var_name
        os.environ[self.env_var_name] = "True"
 
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, False)
        os.environ[self.env_var_name] = "False"


class PositiveArgAction(argparse.Action):
    def __init__(self, option_strings, dest, env_var_name, **kwargs):
        super(PositiveArgAction, self).__init__(option_strings, dest, nargs=0, default=True, **kwargs)
        self.env_var_name = env_var_name
        os.environ[self.env_var_name] = "False"
 
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, False)
        os.environ[self.env_var_name] = "True"


# Set random seed for reproducibility.
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    flow.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('bs', type=int)
parser.add_argument('threshold', type=str)
parser.add_argument('iters', type=int)
parser.add_argument('exp_id', type=str)
parser.add_argument('--no-dtr', action=NegativeArgAction, env_var_name="OF_DTR")
parser.add_argument('--no-lr', action=NegativeArgAction, env_var_name="OF_DTR_LR")
parser.add_argument('--no-o-one', action=NegativeArgAction, env_var_name="OF_DTR_O_ONE")
parser.add_argument('--no-ee', action=NegativeArgAction, env_var_name="OF_DTR_EE")
parser.add_argument('--no-allocator', action=NegativeArgAction, env_var_name="OF_DTR_ALLO")
parser.add_argument('--nlr', action=PositiveArgAction, env_var_name="OF_DTR_NLR")
parser.add_argument('--high-conv', action=PositiveArgAction, env_var_name="OF_DTR_HIGH_CONV")
parser.add_argument('--high-add-n', action=PositiveArgAction, env_var_name="OF_DTR_HIGH_ADD_N")
parser.add_argument('--debug-level', type=int, default=0)

args = parser.parse_args()

print(os.environ)

import oneflow as flow
import oneflow.nn as nn
import flowvision
import torch
import torchvision

# import resnet50_model
import resnet50

# run forward, backward and update parameters
WARMUP_ITERS = 5
ALL_ITERS = args.iters

# NOTE: it has not effect for dtr allocator
heuristic = "eq"

if args.dtr:
    print(f'dtr_enabled: {args.dtr}, dtr_allo: {args.allocator}, threshold: {args.threshold}, batch size: {args.bs}, eager eviction: {args.ee}, left and right: {args.lr}, debug_level: {args.debug_level}, heuristic: {heuristic}, o_one: {args.o_one}')
else:
    print(f'dtr_enabled: {args.dtr}')

if args.dtr:
    flow.enable_dtr(args.dtr, args.threshold, args.debug_level, heuristic)

seed = 20
setup_seed(seed)

writer = SummaryWriter('./tensorboard/' + args.exp_id)

def display():
    flow._oneflow_internal.dtr.display()


# init model
# model = resnet50_model.resnet50(norm_layer=nn.Identity)
model = resnet50.resnet50()
# model = resnet50_model.resnet50()
# model.load_state_dict(flow.load('/tmp/abcde'))
# flow.save(model.state_dict(), '/tmp/abcde')

criterion = nn.CrossEntropyLoss()

cuda0 = flow.device('cuda:1')
sync_tensor = flow.tensor([1, 2, 3]).to(cuda0)


# enable module to use cuda
model.to(cuda0)

criterion.to(cuda0)

learning_rate = 1e-3
# optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

# no bn (only activation memory pool):
# full: 1700MB
# lr=1 ee=0 high add_n cost 600mb success
# lr=1 ee=1 high add_n cost 600mb fail

# has bn:
# full: 2850mb 0.250s
# lr=1 ee=1 high add_n cost 850mb 0.352s
# lr=1 ee=0 high add_n cost 850mb 0.477s
# lr=1 ee=0 high add_n cost 750mb 0.528s

# max threshold 7700mb
# no dtr: bs 80 0.631s
# nlr=1 ee=1 bs 120 0.985s 1.56x
# lr=1 ee=1 high add_n cost bs 280 2.59s 4.10x
# lr=1 ee=1 high add_n cost bs 240 2.22s 3.52x
# lr=0 ee=1 bs 160 1.34s
# lr=1 ee=1 bs 160 1.36s
# lr=1 ee=1 high add_n cost bs 160 1.35s 2.19x
# lr=1 ee=1 high add_n conv cost bs 160 1.42s 2.25x
# lr=0 ee=1 high add_n conv cost bs 160 1.45s 2.30x

# new_lr=1 new_ee=1 bs 120 0.98?s

# ------ new focal docker

# max threshold 7650mb
# no dtr: bs 80 0.631s
# nlr=1 ee=1 bs 120 ?s ?x
# lr=1 ee=1 bs 120 0.991s 1.57x
# lr=1 ee=1 normal add_n cost bs 160 1.45s 2.30x
# lr=1 ee=1 high add_n cost bs 160 1.44s 2.28x
# lr=1 ee=1 normal add_n cost bs 240 2.25s 3.57x
# lr=0 ee=1 normal add_n cost bs 240 2.26s ?x
# lr=0 ee=1 high add_n cost bs 240 2.26s ?x
# lr=1 ee=1 high add_n cost bs 240 2.24s 3.55x
# lr=1 ee=0 high add_n cost bs 280 4.11s 6.51x
# nlr=1 ee=1 bs 160 ?s ?x

# ---

# lr=1 ee=0 threshold 3800mb 0.72s
# lr=1 ee=0 threshold 3500mb 0.70s

# generate random data and label
train_data = flow.tensor(
    np.random.uniform(size=(args.bs, 3, 224, 224)).astype(np.float32), device=cuda0
)
train_label = flow.tensor(
    (np.random.uniform(size=(args.bs,)) * 1000).astype(np.int32), dtype=flow.int32, device=cuda0
)

# load tiny-imagenet dataset
transforms = flowvision.transforms.Compose([flowvision.transforms.Resize(224),flowvision.transforms.ToTensor()])
imagenet_data = flowvision.datasets.ImageFolder('/dataset/tiny-imagenet-200/train/', transforms)
# imagenet_data = flowvision.datasets.ImageFolder('/dataset/tiny-imagenet-200/train/', flowvision.transforms.ToTensor())
train_data_loader = flow.utils.data.DataLoader(imagenet_data, batch_size=args.bs, shuffle=False, num_workers=0)

total_time = 0
for iter in range(ALL_ITERS):
    if args.dtr:
        for x in model.parameters():
            x.grad = flow.zeros_like(x).to(cuda0)

        flow.comm.barrier()
        # temp()
    if iter >= WARMUP_ITERS:
        start_time = time.time()
    
    train_bar = tqdm(train_data_loader, dynamic_ncols=True)

    loss_logger = DataLogger()

    for train_data, train_label in train_bar:
        train_data = train_data.to(cuda0)
        train_label = train_label.to(cuda0)
        logits = model(train_data)
        loss = criterion(logits, train_label)
        loss_logger.update(loss.item(), args.bs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(True)

        train_bar.set_description(
            'Epoch {}: loss: {:.4f}'.format(iter + 1, loss_logger.avg)
        )

    writer.add_scalar('Loss/train/loss', loss_logger.avg, iter)

    del logits
    del loss
    flow.comm.barrier()
    # sync()
    if iter >= WARMUP_ITERS:
        end_time = time.time()
        this_time = end_time - start_time
    #     print(f'iter: {iter}, time: {this_time}')
        total_time += this_time
    # print(f'iter {iter} end')

end_time = time.time()
print(f'{ALL_ITERS - WARMUP_ITERS} iters: avg {(total_time) / (ALL_ITERS - WARMUP_ITERS)}s')
