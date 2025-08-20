"""
Copyright 2021 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from posixpath import join
import unittest
import os
import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from oneflow.nn.parallel import DistributedDataParallel as ddp
import oneflow.unittest
from collections import OrderedDict
from test_util import GenArgDict
import requests
import shutil
from easydict import EasyDict as edict


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.ReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
    ):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.ReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion *
                            self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05,),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = flow.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)

        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet(
        "iresnet18", IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def get_model(name, **kwargs):
    if name == "r18":
        return iresnet18(False, **kwargs)
    else:
        raise ValueError()


def prepare(base_path="/insightface_dataset/"):
    file_url = 'https://oneflow-test.oss-cn-beijing.aliyuncs.com/models_test/insightface_ci.zip'
    # down data
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        res = requests.get(url=file_url)
        with open(os.path.join(base_path, "insightface_ci.zip"), mode='wb') as f:
            f.write(res.content)
        shutil.unpack_archive(filename=os.path.join(
            base_path, "insightface_ci.zip"), extract_dir=base_path, format='zip')
    else:
        if not os.path.exists(os.path.join(base_path, "insightface_ci/")):
            res = requests.get(url=file_url)
            with open(os.path.join(base_path, "insightface_ci.zip"), mode='wb') as f:
                f.write(res.content)
            shutil.unpack_archive(filename=os.path.join(
                base_path, "insightface_ci.zip"), extract_dir=base_path, format='zip')


class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        ofrecord_root: str = "./ofrecord",
        mode: str = "train",  # "val"
        dataset_size: int = 9469,
        batch_size: int = 1,
        total_batch_size: int = 1,
        data_part_num: int = 8,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()
        channel_last = False
        output_layout = "NHWC" if channel_last else "NCHW"
        assert (ofrecord_root, mode)
        self.train_record_reader = flow.nn.OfrecordReader(
            os.path.join(ofrecord_root, mode),
            batch_size=batch_size,
            data_part_num=data_part_num,
            part_name_suffix_length=5,
            random_shuffle=False,
            shuffle_after_epoch=False,
            placement=placement,
            sbp=sbp,
        )
        self.record_label_decoder = flow.nn.OfrecordRawDecoder(
            "label", shape=(), dtype=flow.int32
        )

        color_space = "RGB"
        height = 112
        width = 112

        self.record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )
        self.resize = (
            flow.nn.image.Resize(target_size=[height, width])
            if mode == "train"
            else flow.nn.image.Resize(
                resize_side="shorter", keep_aspect_ratio=True, target_size=112
            )
        )

        self.flip = None
        

        rgb_mean = [127.5, 127.5, 127.5]
        rgb_std = [127.5, 127.5, 127.5]
        self.crop_mirror_norm = (
            flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            ))
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.total_batch_size

    def forward(self):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        image_raw_buffer = self.record_image_decoder(train_record)

        image = self.resize(image_raw_buffer)[0]

        rng = self.flip() if self.flip != None else None
        image = self.crop_mirror_norm(image, rng)

        return image, label


def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )


def meter(self, mkey, *args):
    assert mkey in self.m
    self.m[mkey]["meter"].record(*args)


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        cfg,
        combine_margin,
        cross_entropy,
        data_loader,
        optimizer,
        lr_scheduler=None,
    ):
        super().__init__()

        if cfg.fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(make_grad_scaler())
        elif cfg.scale_grad:
            self.set_grad_scaler(make_static_grad_scaler())

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.model = model

        self.cross_entropy = cross_entropy
        self.combine_margin = combine_margin
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        image, label = self.data_loader()

        image = image.to("cuda")
        label = label.to("cuda")

        logits, label = self.model(image, label)
        logits = self.combine_margin(logits, label) * 64
        loss = self.cross_entropy(logits, label)

        loss.backward()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FC7(flow.nn.Module):
    def __init__(self, embedding_size, num_classes, cfg, partial_fc=False, bias=False):
        super(FC7, self).__init__()
        self.weight = flow.nn.Parameter(
            flow.empty(num_classes, embedding_size))
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)

        self.partial_fc = partial_fc

        size = flow.env.get_world_size()
        num_local = (cfg.num_classes + size - 1) // size
        self.num_sample = int(num_local * cfg.sample_rate)
        self.total_num_sample = self.num_sample * size

    def forward(self, x, label):
        x = flow.nn.functional.l2_normalize(x, 1, epsilon=1e-10)
        if self.partial_fc:
            (
                mapped_label,
                sampled_label,
                sampled_weight,
            ) = flow.distributed_partial_fc_sample(
                weight=self.weight, label=label, num_sample=self.total_num_sample,
            )
            label = mapped_label
            weight = sampled_weight
        else:
            weight = self.weight
        weight = flow.nn.functional.l2_normalize(
            weight, 1, epsilon=1e-10)
        x = flow.matmul(x, weight, transpose_b=True)
        if x.is_consistent:
            return x, label
        else:
            return x


def make_data_loader(args, mode, is_consistent=False, synthetic=False):
    assert mode in ("train", "validation")

    if mode == "train":
        total_batch_size = args.batch_size*flow.env.get_world_size()
        batch_size = args.batch_size
        num_samples = args.num_image
    else:
        total_batch_size = args.val_global_batch_size
        batch_size = args.val_batch_size
        num_samples = args.val_samples_per_epoch

    placement = None
    sbp = None

    if is_consistent:
        placement = flow.env.all_device_placement("cpu")
        sbp = flow.sbp.split(0)
        batch_size = total_batch_size

    ofrecord_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode=mode,
        dataset_size=num_samples,
        batch_size=batch_size,
        total_batch_size=total_batch_size,
        data_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )
    return ofrecord_data_loader


def make_optimizer(args, model):
    param_group = {"params": [p for p in model.parameters() if p is not None]}

    optimizer = flow.optim.SGD(
        [param_group],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer


class Train_Module(flow.nn.Module):
    def __init__(self, cfg, backbone, placement, world_size):
        super(Train_Module, self).__init__()
        self.placement = placement

        if cfg.graph:
            if cfg.model_parallel:
                input_size = cfg.embedding_size
                output_size = int(cfg.num_classes/world_size)
                self.fc = FC7(input_size, output_size, cfg, partial_fc=cfg.partial_fc).to_consistent(
                    placement=placement, sbp=flow.sbp.split(0))
            else:
                self.fc = FC7(cfg.embedding_size, cfg.num_classes, cfg).to_consistent(
                    placement=placement, sbp=flow.sbp.broadcast)
            self.backbone = backbone.to_consistent(
                placement=placement, sbp=flow.sbp.broadcast)
        else:
            self.backbone = backbone
            self.fc = FC7(cfg.embedding_size, cfg.num_classes, cfg)

    def forward(self, x, labels):
        x = self.backbone(x)
        if x.is_consistent:
            x = x.to_consistent(sbp=flow.sbp.broadcast)
        x = self.fc(x, labels)
        return x


class Trainer(object):
    def __init__(self, cfg, placement, load_path, world_size, rank, train_total_steps=1000):
        self.placement = placement
        self.load_path = load_path
        self.cfg = cfg
        self.world_size = world_size
        self.rank = rank
        self.train_total_steps = train_total_steps
        self.frequent = 50

        # model
        self.backbone = get_model(cfg.network, dropout=0.0,
                                  num_features=cfg.embedding_size).to("cuda")
        self.train_module = Train_Module(
            cfg, self.backbone, self.placement, world_size).to("cuda")
        if cfg.resume:
            if load_path is not None:
                self.load_state_dict()
            else:
                print("Model resume failed! load path is None ")

        # optimizer
        self.optimizer = make_optimizer(cfg, self.train_module)

        # data
        self.train_data_loader = make_data_loader(
            cfg, 'train', self.cfg.graph, self.cfg.synthetic)

        # loss
        if cfg.loss == "cosface":
            self.margin_softmax = flow.nn.CombinedMarginLoss(
                1, 0., 0.1).to("cuda")
        else:
            self.margin_softmax = flow.nn.CombinedMarginLoss(
                1, 0.5, 0.).to("cuda")

        self.of_cross_entropy = flow.nn.CrossEntropyLoss()
        # lr_scheduler
        self.decay_step = self.cal_decay_step()
        self.scheduler = flow.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=self.decay_step, gamma=0.1
        )

        self.losses = AverageMeter()
        self.start_epoch = 0
        self.global_step = 0

    def __call__(self):
        # Train
        if self.cfg.graph:
            return self.train_graph()
        else:
            return self.train_eager()

    def load_state_dict(self):

        if self.cfg.graph:
            state_dict = flow.load(self.load_path, consistent_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.load_path)
        else:
            return
        print("Model resume successfully!")
        self.train_module.load_state_dict(state_dict)

    def cal_decay_step(self):
        cfg = self.cfg
        num_image = cfg.num_image
        total_batch_size = cfg.batch_size * self.world_size
        self.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
        self.cfg.total_step = num_image // total_batch_size * cfg.num_epoch
        print("Total Step is:%d" % self.cfg.total_step)
        return [x * num_image // total_batch_size for x in cfg.decay_epoch]

    def train_graph(self):
        train_graph = TrainGraph(self.train_module, self.cfg, self.margin_softmax,
                                 self.of_cross_entropy, self.train_data_loader, self.optimizer, self.scheduler)

        self.train_module.train()
        for steps in range(self.train_total_steps):
            self.global_step += 1
            loss = train_graph()
            loss = loss.to_consistent(
                sbp=flow.sbp.broadcast).to_local().numpy()
            self.losses.update(loss, 1)
        return self.losses.avg

    def train_eager(self):
        self.train_module = ddp(self.train_module)

        self.train_module.train()
        for steps in range(self.train_total_steps):
            self.global_step += 1
            image, label = self.train_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            features_fc7 = self.train_module(image, label)
            features_fc7 = self.margin_softmax(features_fc7, label)*64
            loss = self.of_cross_entropy(features_fc7, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss = loss.numpy()
            self.losses.update(loss, 1)

        return self.losses.avg


def train(test_case, graph, resume, fp16):
    work_path = "/insightface_dataset/"
    prepare(work_path)
    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()
    placement = flow.env.all_device_placement("cuda")
    load_path = os.path.join(work_path, "insightface_ci", "base_model")
    total_batch_size = 64
    loss_base=16

    cfg = edict()
    cfg.batch_size = int(total_batch_size / world_size)
    cfg.loss = "cosface"
    cfg.graph = graph
    cfg.resume = resume
    cfg.fp16 = fp16
    cfg.network = "r18"
    cfg.output = None
    cfg.embedding_size = 512
    cfg.model_parallel = False
    cfg.partial_fc = 0
    cfg.sample_rate = 1
    cfg.momentum = 0.9
    cfg.weight_decay = 5e-4
    cfg.lr = 0.1
    cfg.synthetic = False
    cfg.scale_grad = False
    cfg.dataset = "Agedb"
    cfg.ofrecord_path = "/data/disk1/zhuwang/face_data/data_oneflow/"
    cfg.ofrecord_part_num = 4
    cfg.num_classes = 564
    cfg.num_image = 16483
    cfg.num_epoch = 100
    cfg.warmup_epoch = -1
    cfg.decay_epoch = [80, 90]

    trainer = Trainer(cfg, placement, load_path, world_size, rank, 500)
    loss = trainer()
    test_case.assertTrue(abs(loss-loss_base)<1)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestInsightfaceTrain(flow.unittest.TestCase):
    def test_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["graph"] = [True]
        arg_dict["resume"] = [True]
        arg_dict["fp16"] = [True, False]
        for arg in GenArgDict(arg_dict):
            train(test_case, **arg)

    def test_eager(test_case):
        arg_dict = OrderedDict()
        arg_dict["graph"] = [False]
        arg_dict["resume"] = [True]
        arg_dict["fp16"] = [False]
        for arg in GenArgDict(arg_dict):
            train(test_case,  **arg)


if __name__ == "__main__":
    unittest.main()
