# coding=utf-8

import argparse
import time
import os
import json

import numpy as np
from numpy import random
from torch.utils.tensorboard import SummaryWriter


class NegativeArgAction(argparse.Action):
    def __init__(self, option_strings, dest, env_var_name, **kwargs):
        assert len(option_strings) == 1
        assert "--no-" in option_strings[0]
        dest = dest[3:]
        super(NegativeArgAction, self).__init__(
            # default value is True
            option_strings,
            dest,
            nargs=0,
            default=True,
            **kwargs,
        )
        self.env_var_name = env_var_name
        os.environ[self.env_var_name] = "True"

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, False)
        os.environ[self.env_var_name] = "False"


class PositiveArgAction(argparse.Action):
    def __init__(self, option_strings, dest, env_var_name, **kwargs):
        super(PositiveArgAction, self).__init__(
            # default value is False
            option_strings,
            dest,
            nargs=0,
            default=False,
            **kwargs,
        )
        self.env_var_name = env_var_name
        os.environ[self.env_var_name] = "False"

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        os.environ[self.env_var_name] = "True"


# Set random seed for reproducibility.
def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    flow.manual_seed(seed)


prefix = os.getenv("ONEFLOW_DTR_SUMMARY_FILE_PREFIX")


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name like resnet50, unet etc.")
parser.add_argument("bs", type=int)
parser.add_argument("threshold", type=str)
parser.add_argument("iters", type=int)
parser.add_argument("--exp_id", type=str)
parser.add_argument("--no-dtr", action=NegativeArgAction, env_var_name="OF_DTR")
parser.add_argument("--no-lr", action=NegativeArgAction, env_var_name="OF_DTR_LR")
parser.add_argument(
    "--no-fbip", action=NegativeArgAction, env_var_name="ONEFLOW_DTR_FBIP"
)
parser.add_argument(
    "--old-immutable",
    action=PositiveArgAction,
    env_var_name="ONEFLOW_DTR_OLD_IMMUTABLE",
)
parser.add_argument("--no-o-one", action=NegativeArgAction, env_var_name="OF_DTR_O_ONE")
parser.add_argument("--no-ee", action=NegativeArgAction, env_var_name="OF_DTR_EE")
parser.add_argument(
    "--no-allocator", action=NegativeArgAction, env_var_name="OF_DTR_ALLO"
)
parser.add_argument("--nlr", action=PositiveArgAction, env_var_name="OF_DTR_NLR")
parser.add_argument(
    "--high-conv", action=PositiveArgAction, env_var_name="OF_DTR_HIGH_CONV"
)
parser.add_argument(
    "--high-add-n", action=PositiveArgAction, env_var_name="OF_DTR_HIGH_ADD_N"
)
parser.add_argument("--debug-level", type=int, default=0)
parser.add_argument("--no-dataloader", action="store_true")

args = parser.parse_args()

if args.debug_level > 0:
    print(os.environ)

import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms
import flowvision.models as models

WARMUP_ITERS = 5
ALL_ITERS = args.iters

if args.allocator:
    heuristic = "eq_compute_time_and_last_access"
else:
    heuristic = "eq"

if args.dtr:
    print(
        f"model_name: {args.model_name}, dtr_enabled: {args.dtr}, dtr_allo: {args.allocator}, threshold: {args.threshold}, batch size: {args.bs}, eager eviction: {args.ee}, left and right: {args.lr}, debug_level: {args.debug_level}, heuristic: {heuristic}, o_one: {args.o_one}"
    )
else:
    print(f"model_name: {args.model_name}, dtr_enabled: {args.dtr}")

if args.dtr:
    flow.enable_dtr(args.dtr, args.threshold, args.debug_level, heuristic)

setup_seed(20)

enable_tensorboard = args.exp_id is not None
if enable_tensorboard:
    writer = SummaryWriter("./tensorboard/" + args.exp_id)


def get_imagenet_imagefolder():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return flowvision.datasets.ImageFolder(
        "/dataset/imagenet_folder/train",
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )


def resnet50_info():
    model = models.resnet50()
    criterion = nn.CrossEntropyLoss()
    get_fixed_input = lambda bs: flow.ones(bs, 3, 224, 224)
    get_fixed_label = lambda bs: flow.ones(bs, dtype=flow.int64)
    return model, criterion, get_fixed_input, get_fixed_label, get_imagenet_imagefolder


def densenet121_info():
    model = models.densenet121()
    criterion = nn.CrossEntropyLoss()
    get_fixed_input = lambda bs: flow.ones(bs, 3, 224, 224)
    get_fixed_label = lambda bs: flow.ones(bs, dtype=flow.int64)
    return model, criterion, get_fixed_input, get_fixed_label, get_imagenet_imagefolder


def unet_info():
    import unet

    model = unet.UNet(n_channels=3, n_classes=2, bilinear=False)
    criterion = nn.BCEWithLogitsLoss()
    get_fixed_input = lambda bs: flow.ones(bs, 3, 460, 608)
    get_fixed_label = lambda bs: flow.ones(bs, 2, 460, 608)

    def get_imagefolder():
        raise NotImplementedError

    return model, criterion, get_fixed_input, get_fixed_label, get_imagefolder


model, criterion, get_fixed_input, get_fixed_label, get_imagefolder = eval(
    f"{args.model_name}_info()"
)

model.to("cuda")
criterion.to("cuda")

learning_rate = 0.001
optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)


if args.no_dataloader:
    global data
    global label
    data = get_fixed_input(args.bs)
    label = get_fixed_label(args.bs)

    class FixedDataset(flow.utils.data.Dataset):
        def __len__(self):
            return 999999999

        def __getitem__(self, idx):
            return data, label

    train_data_loader = FixedDataset()
else:
    imagefolder = get_imagefolder()
    train_data_loader = flow.utils.data.DataLoader(
        imagefolder, batch_size=args.bs, shuffle=True, num_workers=0
    )

total_time = 0

if args.dtr:
    flow.nn.ContiguousGrad(model)
zero_grad_set_to_none = args.old_immutable

for iter, (train_data, train_label) in enumerate(train_data_loader):
    if iter >= ALL_ITERS:
        break

    train_data = train_data.to("cuda")
    train_label = train_label.to("cuda")

    flow.comm.barrier()
    # print(f'iter {iter} start, all pieces:')
    # flow._oneflow_internal.dtr.display_all_pieces()

    logits = model(train_data)
    loss = criterion(logits, train_label)
    loss.backward()
    if enable_tensorboard:
        writer.add_scalar("Loss/train/loss", loss.item(), iter)
        writer.flush()

    optimizer.step()
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
    del logits
    del loss

    flow.comm.barrier()

    if iter >= WARMUP_ITERS:
        this_time = time.time() - last_time
        total_time += this_time
        if iter % 1 == 0:
            pass
            # print(f'iter {iter} end, time: {this_time}')

    last_time = time.time()
    if iter == 0:
        pass
        # print('iter 0 ok')
    # print(f'iter {iter} end, all pieces:')
    # flow._oneflow_internal.dtr.display_all_pieces()
    flow._oneflow_internal.dtr.set_left(True)

time_per_run = total_time / (ALL_ITERS - WARMUP_ITERS)
print(f"{ALL_ITERS - WARMUP_ITERS} iters: avg {time_per_run}s")

if prefix is not None:
    fn = f"{prefix}.json"
    with open(fn, "w") as f:
        json.dump(
            {
                "real time": time_per_run,
                "threshold": args.threshold,
                "model_name": args.model_name,
                "batch_size": args.bs,
            },
            f,
        )

