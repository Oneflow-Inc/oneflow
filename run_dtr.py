# coding=utf-8

import argparse
import time
import os
import json

import numpy as np
from numpy import random


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
    "--me-style", action=PositiveArgAction, env_var_name="ONEFLOW_DTR_MEGENGINE_STYLE"
)
parser.add_argument(
    "--with-size",
    action=PositiveArgAction,
    env_var_name="ONEFLOW_DTR_HEURISTIC_WITH_SIZE",
)
parser.add_argument(
    "--high-conv", action=PositiveArgAction, env_var_name="OF_DTR_HIGH_CONV"
)
parser.add_argument(
    "--high-add-n", action=PositiveArgAction, env_var_name="OF_DTR_HIGH_ADD_N"
)
parser.add_argument("--group-num", type=int, required=True)
parser.add_argument("--debug-level", type=int, default=0)
parser.add_argument("--me-method", type=str, default="eq")
parser.add_argument("--no-dataloader", action="store_true")

args = parser.parse_args()
assert not (args.me_style and args.with_size)

os.environ["ONEFLOW_DTR_GROUP_NUM"] = str(args.group_num)

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
    # heuristic = "full_compute_time_and_last_access"
    if args.me_style:
        heuristic = args.me_method

else:
    heuristic = "eq"

if args.dtr:
    print(
        f"model_name: {args.model_name}, dtr_enabled: {args.dtr}, dtr_allo: {args.allocator}, threshold: {args.threshold}, batch size: {args.bs}, eager eviction: {args.ee}, left and right: {args.lr}, debug_level: {args.debug_level}, heuristic: {heuristic}, o_one: {args.o_one}, me_style: {args.me_style}, with_size: {args.with_size}, group_num: {args.group_num}"
    )
else:
    print(f"model_name: {args.model_name}, dtr_enabled: {args.dtr}")

if args.nlr and not args.lr:
    raise ValueError()

if args.dtr:
    flow.enable_dtr(args.dtr, args.threshold, args.debug_level, heuristic)

setup_seed(20)

enable_tensorboard = args.exp_id is not None
if enable_tensorboard:
    from torch.utils.tensorboard import SummaryWriter

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


def resnet152_info():
    model = models.resnet152()
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


def swin_transformer_info():
    model = models.swin_base_patch4_window7_224()
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


def update_dataset(prof):
    DATASET_FILENAME = f"/home/dev/{args.model_name}_op_time.json"
    if os.path.exists(DATASET_FILENAME):
        with open(DATASET_FILENAME, "r") as f:
            time_dict = json.load(f)
    else:
        time_dict = {}

    events = prof.key_averages()
    new_time_dict = {}
    for e in events:
        if isinstance(e, flow.profiler.events.KernelEvent):
            new_time_dict[
                f"{e.name} {e.description['shape']} {e.description['attr']}"
            ] = (e.cuda_time_total, e.count)

    time_dict.update(new_time_dict)

    with open(DATASET_FILENAME, "w") as f:
        json.dump(time_dict, f)


class Nothing:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


PROFILE = False

if PROFILE:
    profile_guard = lambda: flow.profiler.profile(record_shapes=True)
else:
    profile_guard = Nothing

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
    data = get_fixed_input(args.bs).to("cuda")
    label = get_fixed_label(args.bs).to("cuda")

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

if args.dtr:
    flow.nn.ContiguousGrad(model)
zero_grad_set_to_none = args.old_immutable


def run_iter(train_data, train_label):
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
    if args.allocator:
        flow._oneflow_internal.dtr.set_left(True)


for iter, (train_data, train_label) in enumerate(train_data_loader):
    if iter >= WARMUP_ITERS or iter >= ALL_ITERS:
        break

    run_iter(train_data, train_label)


total_time = 0
SKIP_FIRST_ITERS = 2

with profile_guard() as prof:
    last_time = time.time()

    for iter, (train_data, train_label) in enumerate(train_data_loader):
        if iter >= ALL_ITERS - WARMUP_ITERS:
            break

        run_iter(train_data, train_label)

        this_time = time.time() - last_time
        # Skip iter 0 and 1, whose time is strangely shorter
        if iter >= SKIP_FIRST_ITERS:
            print(f"iter={iter}, time={this_time}")
            total_time += this_time
        last_time = time.time()

if PROFILE:
    print(prof.key_averages())
    update_dataset(prof)

time_per_run = total_time / (ALL_ITERS - WARMUP_ITERS - SKIP_FIRST_ITERS)
print(f"{ALL_ITERS - WARMUP_ITERS - SKIP_FIRST_ITERS} iters: avg {time_per_run}s")

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

