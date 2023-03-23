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


def compare_with_numpy_rmsprop(
    test_case,
    device,
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
    contiguous_params,
    check_allclose=True,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
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
                    "contiguous_params": contiguous_params,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad,
                dtype=flow.float32,
                requires_grad=False,
                device=flow.device(device),
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            rmsprop.step()
            rmsprop.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = rmsprop.state_dict()
                rmsprop = flow.optim.RMSprop([x], contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                rmsprop.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
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
            (x, r, g, v) = train_one_iter(random_grad_seq[i])
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
    device,
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
    contiguous_params,
    check_allclose=True,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
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
                    "contiguous_params": contiguous_params,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad,
                dtype=flow.float32,
                requires_grad=False,
                device=flow.device(device),
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            rmsprop.clip_grad()
            rmsprop.step()
            rmsprop.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = rmsprop.state_dict()
                rmsprop = flow.optim.RMSprop([x], contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                rmsprop.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
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
            (x, r, g, v) = train_one_iter(random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    if check_allclose:
        test_case.assertTrue(
            np.allclose(
                oneflow_res.flatten(), numpy_res.flatten(), rtol=2e-3, atol=2e-3
            )
        )


@flow.unittest.skip_unless_1n1d()
class TestRMSProp(flow.unittest.TestCase):
    def test_rmsprop(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1]
        arg_dict["momentum"] = [0.0]
        arg_dict["train_iters"] = [2]
        arg_dict["alpha"] = [0.9, 0.99]
        arg_dict["eps"] = [1e-08, 1e-05]
        arg_dict["weight_decay"] = [0.1, 0.99]
        arg_dict["centered"] = [False, True]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["contiguous_params"] = [True, False]
        arg_dict["check_allclose"] = [False]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_rmsprop(test_case, *arg)

    def test_rmsprop_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1]
        arg_dict["momentum"] = [0.0]
        arg_dict["train_iters"] = [2]
        arg_dict["alpha"] = [0.9, 0.99]
        arg_dict["eps"] = [1e-08, 1e-05]
        arg_dict["weight_decay"] = [0.1, 0.99]
        arg_dict["centered"] = [False, True]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["contiguous_params"] = [False, True]
        arg_dict["check_allclose"] = [False]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_rmsprop_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
