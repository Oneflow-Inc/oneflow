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


def compare_with_numpy_adadelta(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    rho,
    eps,
    maximize,
    weight_decay,
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
        adadelta = flow.optim.Adadelta(
            [{"params": [x], "lr": learning_rate, "weight_decay": weight_decay,}],
            rho=rho,
            eps=eps,
            maximize=maximize,
            contiguous_params=contiguous_params,
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adadelta.step()
            adadelta.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adadelta.state_dict()
                adadelta = flow.optim.Adadelta([x], contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                adadelta.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        square_avgs = np.zeros_like(x)
        acc_deltas = np.zeros_like(x)

        def train_one_iter(grad):
            grad = grad if not maximize else -grad
            grad = grad + weight_decay * x
            new_square_avgs = square_avgs * rho + (1.0 - rho) * grad * grad
            std = np.sqrt(new_square_avgs + eps)
            delta = np.sqrt(acc_deltas + eps) / std * grad
            new_acc_deltas = acc_deltas * rho + delta * delta * (1 - rho)
            param = x - learning_rate * delta
            return (param, new_square_avgs, new_acc_deltas)

        for i in range(1, train_iters + 1):
            (x, square_avgs, acc_deltas) = train_one_iter(random_grad_seq[i - 1])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()

    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-4, atol=1e-4)
    )


def compare_with_numpy_adadelta_clip_grad(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    rho,
    eps,
    maximize,
    weight_decay,
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
        adadelta = flow.optim.Adadelta(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "clip_grad_max_norm": clip_grad_max_norm,
                    "clip_grad_norm_type": clip_grad_norm_type,
                }
            ],
            rho=rho,
            eps=eps,
            maximize=maximize,
            contiguous_params=contiguous_params,
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adadelta.clip_grad()
            adadelta.step()
            adadelta.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adadelta.state_dict()
                adadelta = flow.optim.Adadelta([x], contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                adadelta.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        square_avgs = np.zeros_like(x)
        acc_deltas = np.zeros_like(x)

        def train_one_iter(grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad if not maximize else -grad
            grad = grad + weight_decay * x
            new_square_avgs = square_avgs * rho + (1.0 - rho) * grad * grad
            std = np.sqrt(new_square_avgs + eps)
            delta = np.sqrt(acc_deltas + eps) / std * grad
            new_acc_deltas = acc_deltas * rho + delta * delta * (1 - rho)
            param = x - learning_rate * delta
            return (param, new_square_avgs, new_acc_deltas)

        for i in range(1, train_iters + 1):
            (x, square_avgs, acc_deltas) = train_one_iter(random_grad_seq[i - 1])

        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()

    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-4, atol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
class TestAdadelta(flow.unittest.TestCase):
    def test_adadelta(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["rho"] = [0.9, 0.6]
        arg_dict["eps"] = [1e-6, 1e-4]
        arg_dict["maximize"] = [False]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["contiguous_params"] = [False, True]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adadelta(test_case, *arg)

    def test_adadelta_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["rho"] = [0.9, 0.6]
        arg_dict["eps"] = [1e-6, 1e-4]
        arg_dict["maximize"] = [False]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        arg_dict["contiguous_params"] = [False, True]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adadelta_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
