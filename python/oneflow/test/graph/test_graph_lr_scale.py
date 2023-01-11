"""
Copyright 2020 The OneFlow Authors. All rights reserved.

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
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest

from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


class _Block(flow.nn.Module):
    def __init__(self, feats, device=None, placement=None):
        super().__init__()
        ones = flow.ones(feats)
        if placement is not None:
            ones = ones.to_global(placement=placement, sbp=flow.sbp.broadcast())
        elif device is not None:
            ones = ones.to(device)
        self.param = flow.nn.Parameter(ones)

    def forward(self, x):
        return x + self.param


class _MyModule(flow.nn.Module):
    def __init__(self, feats, depth, device=None, placement=None):
        super().__init__()
        self.layers = flow.nn.ModuleList(
            [
                _Block(feats=feats, device=device, placement=placement)
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _MyGraph(flow.nn.Graph):
    def __init__(self, model, optimizer, lr_scheduler):
        super().__init__()
        self.m = model
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self, input):
        out = self.m(input)
        out.sum().backward()
        return out


def _lrs_param_groups(model, base_scale):
    param_groups = []
    for i, layer in enumerate(model.layers):
        this_scale = base_scale ** (i + 1)
        param_group = {"params": layer.parameters(), "lr_scale": this_scale}
        param_groups.append(param_group)

    return param_groups


def _rand_input(shape, device=None, placement=None, requires_grad=False):
    input = flow.tensor(np.random.rand(*shape).astype(np.float32))
    if placement is not None:
        input = input.to_global(placement=placement, sbp=flow.sbp.split(0))
    elif device is not None:
        input = input.to(device)
    if requires_grad:
        input.requires_grad_()
    return input


def _test_lrs(test_case, **kwargs):
    verbose = kwargs.pop("verbose", False)
    if verbose:
        print(f"#### kwargs={kwargs}")

    batch_size = kwargs.pop("batch_size", 4)
    feats = kwargs.pop("feats", 768)
    depth = kwargs.pop("depth", 3)
    lr = kwargs.pop("lr", 1.0)
    base_scale = kwargs.pop("base_scale", 0.1)
    device_type = kwargs.pop("device_type", "cuda")
    placement = kwargs.pop("placement", None)
    graph_mode = kwargs.pop("graph_mode", True)

    model = _MyModule(feats=feats, depth=depth, device=device_type, placement=placement)
    param_groups = _lrs_param_groups(model, base_scale=base_scale)
    optimizer = flow.optim.SGD(param_groups, lr=lr)
    lr_scheduler = flow.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=100
    )
    model_graph = _MyGraph(model, optimizer, lr_scheduler)

    input = _rand_input(
        (batch_size, feats), device=device_type, placement=placement, requires_grad=True
    )
    t_params = []

    if graph_mode:
        for i in range(depth):
            origin_p = model.layers[i].param.numpy()
            init_grad = float(batch_size * flow.env.get_world_size())
            t_params.append(origin_p - float(init_grad) * lr * (base_scale ** (i + 1)))
        ret = model_graph(input)
    else:
        for i in range(depth):
            origin_p = model.layers[i].param.numpy()
            init_grad = float(batch_size * flow.env.get_world_size())
            t_params.append(origin_p - float(init_grad) * lr)

        optimizer.zero_grad()
        ret = model(input)
        ret.sum().backward()
        optimizer.step()
        lr_scheduler.step()

    if verbose:
        print("#### input")
        print(input)
        # sync
        np_ret = ret.numpy()
        print("#### ret")
        print(np_ret)

        for i in range(depth):
            np_param = model.layers[i].param.numpy()
            print(f"#### layer{i} param")
            print(np_param)

        print("#### grad")
        print(input.grad)

    for i in range(depth):
        np_param = model.layers[i].param.numpy()
        t_param = t_params[i]
        test_case.assertTrue(
            np.allclose(np_param, t_param), f"\n{np_param}\n vs. \n{t_param}"
        )


@flow.unittest.skip_unless_1n1d()
class LRScaleTest(flow.unittest.TestCase):
    def test_lr_scale(self):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [2, 4]
        arg_dict["feats"] = [10, 13]
        arg_dict["depth"] = [3, 4]
        arg_dict["lr"] = [1.0, 0.1]
        arg_dict["base_scale"] = [0.1, 0.2]
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["is_global"] = [True, False]
        arg_dict["graph_mode"] = [True, False]

        for arg in GenArgDict(arg_dict):
            is_global = arg.pop("is_global", True)
            if is_global:
                device_type = arg.pop("device_type", "cuda")
                arg["placement"] = flow.placement.all(device_type)

            # arg["verbose"] = True
            _test_lrs(self, **arg)


@flow.unittest.skip_unless_1n2d()
class LRScaleParallelTest(flow.unittest.TestCase):
    def test_lr_scale_parallel(self):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [2, 4]
        arg_dict["feats"] = [5, 10]
        arg_dict["depth"] = [3, 4]
        arg_dict["lr"] = [1.0, 0.1]
        arg_dict["base_scale"] = [0.1, 0.2]
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["graph_mode"] = [True, False]

        for arg in GenArgDict(arg_dict):
            device_type = arg.pop("device_type", "cuda")
            arg["placement"] = flow.placement.all(device_type)
            # arg["verbose"] = True
            _test_lrs(self, **arg)


if __name__ == "__main__":
    unittest.main()
