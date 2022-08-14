import oneflow as flow
import oneflow.nn as nn
from flowvision import datasets
from flowvision import transforms
from oneflow.utils.data import DataLoader
from typing import Any, Type, Union, List, Optional
import time
import argparse
import os
import time
from resnet20 import ResNet
from apex_oneflow import initialize

# learning_rate = 1e-2
learning_rate = 8e-2
batch_size = 1024

print_time = 0
start = 0
def start_tax():
    if not print_time:
        return 
    flow.npu.synchronize()
    global start 
    start = time.time()
def end_tax(str=''):
    if not print_time:
        return 
    flow.npu.synchronize()
    global start
    print(str," ",time.time()-start)
    start = time.time()
def Print(str=''):
    if not print_time:
        return
    print(str)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with flow.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.reshape(batch_size)
        return (pred == target).float().sum().item()

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', nargs='?', default=0, type=int)
    parser.add_argument('--epoch', nargs='?', default=160, type=int)
    parser.add_argument('--is-pretrained', action='store_true')
    args = parser.parse_args()
    return args

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0
    start_time = time.time()
    start_tax()
    for batch, (X, y) in enumerate(dataloader):
        end_tax('pre_data')
        Print(f"===========step {batch}")
        X = X.to('npu').to(flow.float16)
        label = y.to(flow.int32).to('npu')
        pred = model(X)
        end_tax('forward')
        loss = loss_fn(pred, label)
        end_tax('loss')
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        end_tax('backward')
        optimizer.step()
        end_tax('step')
        # if batch==10:
        #     exit()
        correct += accuracy(pred, label)
        end_tax('correct')
    flow.npu.synchronize()
    print(f"Train Error: Accuracy: {(100*correct/size):>0.1f}%")
    end_time = time.time()
    print(f"TimeCost: {(end_time-start_time):>0.1f}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with flow.no_grad():
        for X, y in dataloader:
            # move to gpu
            X = X.to('npu').to(flow.float16)
            label = y.to(flow.int32).to('npu')
            # test
            pred = model(X)
            test_loss += loss_fn(pred, label).item()
            correct += accuracy(pred, label)

    test_loss /= size
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")

def set_tag(s, t):
    time.sleep(t)
    print(s, 't'+os.popen('date +"%s.%N"','r').read().strip())
    time.sleep(t)


def main():


    args = parse_arg()

    model = ResNet(3)
    
    model = model.to('npu')
    model = initialize(model)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.CIFAR10(
        root="/data",
        train=True,
        download=True,
        transform=transform_train
    )
    test_data = datasets.CIFAR10(
        root="/data",
        train=False,
        download=True,
        transform=transform_test
    )
    parallel_workers = 8
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                    num_workers=parallel_workers)#, persistent_workers=(parallel_workers > 0))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                    num_workers=parallel_workers)#, persistent_workers=(parallel_workers > 0))

    set_tag('start loss function and opt :', 0.3)
    loss_fn = nn.CrossEntropyLoss(reduction="mean").to('npu')
    optimizer = flow.optim.TORCH_SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = flow.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    set_tag('finish loss function and opt construct :', 0.3)

    for t in range(args.epoch):
        print(f"Epoch {t+1}:")
        print("-------------------------------------------")
        print("train")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        #print("test")
        lr_scheduler.step()
        if t%10==0:
            test_loop(test_dataloader, model, loss_fn)
    
    set_tag("Done:", 0.3)

if __name__ == '__main__':
    main()
