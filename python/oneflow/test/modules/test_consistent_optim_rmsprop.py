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

import tempfile
import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter

from oneflow.test_utils.automated_test_util import *


def compare_with_numpy_rmsprop(
    test_case,
    placement,
    sbp,
    x_shape,
    learning_rate,
    momentum,
    train_iters,
    alpha,
    eps,
    weight_decay,
    centered,
    reload_state_step,
    save_load_by_pickle,
    check_allclose=True,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(random_tensor(len(x_shape), *x_shape).oneflow)
    init_value = random_tensor(len(x_shape), *x_shape).oneflow

    def train_by_oneflow():
        x = Parameter(init_value.clone().to_global(placement=placement, sbp=sbp))
        param_list = list()
        param_list.append(x)
        rmsprop = flow.optim.RMSprop(
            [
                {
                    "params": param_list,
                    "lr": learning_rate,
                    "alpha": alpha,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "centered": centered,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = grad.clone().to_global(placement, sbp)
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            rmsprop.step()
            rmsprop.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = rmsprop.state_dict()
                rmsprop = flow.optim.RMSprop([x])
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir, global_dst_rank=0)
                        state_dict = flow.load(save_dir, global_src_rank=0)
                rmsprop.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value.numpy()
        r = np.zeros_like(x)
        v = np.zeros_like(x)
        g = np.zeros_like(x)

        def train_one_iter(grad):

            grad = grad + weight_decay * x
            r_ = alpha * r + (1 - alpha) * grad * grad
            if centered:
                g_ = alpha * g + (1 - alpha) * grad
                v_ = momentum * v + learning_rate / np.sqrt(r_ - g_ * g_ + eps) * grad
            else:
                g_ = g
                v_ = momentum * v + learning_rate / np.sqrt(r_ + eps) * grad
            param = x - v_
            return (param, r_, g_, v_)

        for i in range(train_iters):
            (x, r, g, v) = train_one_iter(random_grad_seq[i].numpy())
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    if check_allclose:
        test_case.assertTrue(
            np.allclose(
                oneflow_res.flatten(), numpy_res.flatten(), rtol=2e-3, atol=2e-3
            )
        )


def compare_with_numpy_rmsprop_clip_grad(
    test_case,
    placement,
    sbp,
    x_shape,
    learning_rate,
    momentum,
    train_iters,
    alpha,
    eps,
    weight_decay,
    centered,
    clip_grad_max_norm,
    clip_grad_norm_type,
    reload_state_step,
    save_load_by_pickle,
    check_allclose=True,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(random_tensor(len(x_shape), *x_shape).oneflow)
    init_value = random_tensor(len(x_shape), *x_shape).oneflow

    def train_by_oneflow():
        x = Parameter(init_value.clone().to_global(placement=placement, sbp=sbp))
        param_list = list()
        param_list.append(x)
        rmsprop = flow.optim.RMSprop(
            [
                {
                    "params": param_list,
                    "lr": learning_rate,
                    "alpha": alpha,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "centered": centered,
                    "clip_grad_max_norm": clip_grad_max_norm,
                    "clip_grad_norm_type": clip_grad_norm_type,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = grad.clone().to_global(placement, sbp)
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            rmsprop.clip_grad()
            rmsprop.step()
            rmsprop.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = rmsprop.state_dict()
                rmsprop = flow.optim.RMSprop([x])
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir, global_dst_rank=0)
                        state_dict = flow.load(save_dir, global_src_rank=0)
                rmsprop.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value.numpy()
        r = np.zeros_like(x)
        v = np.zeros_like(x)
        g = np.zeros_like(x)

        def train_one_iter(grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x
            r_ = alpha * r + (1 - alpha) * grad * grad
            if centered:
                g_ = alpha * g + (1 - alpha) * grad
                v_ = momentum * v + learning_rate / np.sqrt(r_ - g_ * g_ + eps) * grad
            else:
                g_ = g
                v_ = momentum * v + learning_rate / np.sqrt(r_ + eps) * grad
            param = x - v_
            return (param, r_, g_, v_)

        for i in range(train_iters):
            (x, r, g, v) = train_one_iter(random_grad_seq[i].numpy())
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    if check_allclose:
        test_case.assertTrue(
            np.allclose(
                oneflow_res.flatten(), numpy_res.flatten(), rtol=2e-3, atol=2e-3
            )
        )


class TestRMSProp(flow.unittest.TestCase):
    @globaltest
    def test_rmsprop(test_case):
        arg_dict = OrderedDict()
        arg_dict["x_shape"] = [(16, 16)]
        arg_dict["learning_rate"] = [1]
        arg_dict["momentum"] = [0.0]
        arg_dict["train_iters"] = [5]
        arg_dict["alpha"] = [0.9]
        arg_dict["eps"] = [1e-05]
        arg_dict["weight_decay"] = [0.99]
        arg_dict["centered"] = [False, True]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["check_allclose"] = [False]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                    compare_with_numpy_rmsprop(test_case, placement, sbp, *arg)

    @globaltest
    def test_rmsprop_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["x_shape"] = [(16, 16)]
        arg_dict["learning_rate"] = [1]
        arg_dict["momentum"] = [0.0]
        arg_dict["train_iters"] = [5]
        arg_dict["alpha"] = [0.9]
        arg_dict["eps"] = [1e-05]
        arg_dict["weight_decay"] = [0.99]
        arg_dict["centered"] = [False, True]
        arg_dict["clip_grad_max_norm"] = [0.5]
        arg_dict["clip_grad_norm_type"] = ["inf", 2.0]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["check_allclose"] = [False]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                    compare_with_numpy_rmsprop_clip_grad(
                        test_case, placement, sbp, *arg
                    )


if __name__ == "__main__":
    unittest.main()
