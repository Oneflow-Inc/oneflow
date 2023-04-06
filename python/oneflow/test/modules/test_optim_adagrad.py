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

import os
import tempfile
import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_adagrad(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    lr_decay,
    weight_decay,
    initial_accumulator_value,
    eps,
    reload_state_step,
    save_load_by_pickle,
    contiguous_params,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        adagrad = flow.optim.Adagrad(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ],
            lr_decay=lr_decay,
            initial_accumulator_value=initial_accumulator_value,
            contiguous_params=contiguous_params,
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adagrad.step()
            adagrad.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adagrad.state_dict()
                adagrad = flow.optim.Adagrad([x], contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                adagrad.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        st = np.ones_like(x) * initial_accumulator_value

        def train_one_iter(iter, grad):
            grad = grad + weight_decay * x
            lr = learning_rate / (1 + (iter - 1) * lr_decay)
            s = st + grad * grad
            param = x - lr / (np.sqrt(s) + eps) * grad
            return (param, s)

        for i in range(1, train_iters + 1):
            (x, st) = train_one_iter(i, random_grad_seq[i - 1])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()

    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-3, atol=1e-3)
    )


def compare_with_numpy_adagrad_clip_grad(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    lr_decay,
    weight_decay,
    initial_accumulator_value,
    eps,
    clip_grad_max_norm,
    clip_grad_norm_type,
    reload_state_step,
    save_load_by_pickle,
    contiguous_params,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        adagrad = flow.optim.Adagrad(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "clip_grad_max_norm": clip_grad_max_norm,
                    "clip_grad_norm_type": clip_grad_norm_type,
                }
            ],
            lr_decay=lr_decay,
            initial_accumulator_value=initial_accumulator_value,
            contiguous_params=contiguous_params,
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adagrad.clip_grad()
            adagrad.step()
            adagrad.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adagrad.state_dict()
                adagrad = flow.optim.Adagrad([x], contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                adagrad.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        st = np.ones_like(x) * initial_accumulator_value

        def train_one_iter(iter, grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x

            lr = learning_rate / (1 + (iter - 1) * lr_decay)
            s = st + grad * grad
            param = x - lr / (np.sqrt(s) + eps) * grad

            return (param, s)

        for i in range(1, train_iters + 1):
            (x, st) = train_one_iter(i, random_grad_seq[i - 1])

        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()

    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-3, atol=1e-3)
    )


@flow.unittest.skip_unless_1n1d()
class TestAdagrad(flow.unittest.TestCase):
    def test_adagrad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["lr_decay"] = [0.9, 0.75]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["initial_accumulator_value"] = [1.0, 2.1]
        arg_dict["eps"] = [1e-08, 1e-07]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["contiguous_params"] = [False, True]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adagrad(test_case, *arg)

    def test_adagrad_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["lr_decay"] = [0.9, 0.75]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["initial_accumulator_value"] = [2.1]
        arg_dict["eps"] = [1e-07]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["contiguous_params"] = [False, True]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adagrad_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
