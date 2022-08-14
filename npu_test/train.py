import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time
import argparse
import os

from resnet import ResNet

# learning_rate = 1e-2
learning_rate = 1e-1
batch_size = 256

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
    for batch, (X, y) in enumerate(dataloader):
        torch.cuda.nvtx.range_push("batch:" + str(batch))
        # move to gpu
        torch.cuda.nvtx.range_push("copy data into device")
        X = X.cuda()
        y = y.cuda()
        torch.cuda.nvtx.range_pop()
        # Compute prediction and loss
        torch.cuda.nvtx.range_push("forward pass")
        pred = model(X)
        torch.cuda.nvtx.range_pop()
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        torch.cuda.nvtx.range_push("backward pass")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("update params")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss/batch_size :>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Train Error: Accuracy: {(100*correct/size):>0.1f}%")
            
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # move to gpu
            X = X.cuda()
            y = y.cuda()
            # test
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")

def set_tag(s, t):
    time.sleep(t)
    print(s, 't'+os.popen('date +"%s.%N"','r').read().strip())
    time.sleep(t)


def main():
    args = parse_arg()
    torch.cuda.set_device(args.device_id)

    set_tag('start model to device :', 0.3)
    # model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=args.is_pretrained)
    # model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = ResNet(3)
    torch.cuda.nvtx.range_push("copy model to device")
    model = model.cuda()
    torch.cuda.nvtx.range_pop()
    set_tag('finish model to device :', 0.3)

    # transform_train = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1/255, 1/255, 1/255])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1/255, 1/255, 1/255])
    ])

    train_data = torchvision.datasets.CIFAR10(
        root="/data",
        train=True,
        download=True,
        transform=transform_train
    )
    test_data = torchvision.datasets.CIFAR10(
        root="/data",
        train=False,
        download=True,
        transform=transform_test
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    set_tag('start loss function and opt :', 0.3)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 0.95 ** epoch))
    set_tag('finish loss function and opt construct :', 0.3)

    for t in range(args.epoch):
        set_tag(f"Epoch {t+1}:", 0.3)
        print("-------------------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        set_tag("Train end:", 0.3)
        lr_scheduler.step()
        set_tag("Test start:", 0.3)
        test_loop(test_dataloader, model, loss_fn)
    
    set_tag("Done:", 0.3)

if __name__ == '__main__':
    main()
