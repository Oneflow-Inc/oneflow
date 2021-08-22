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
from collections import OrderedDict

import numpy as np
from test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_adam(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    betas,
    weight_decay,
    eps,
    do_bias_correction,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        adam = flow.optim.Adam(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ],
            do_bias_correction=do_bias_correction,
        )

        def train_one_iter(grad):
            grad_tensor = flow.Tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def train_one_iter(iter, grad):
            grad = grad + weight_decay * x

            if do_bias_correction:
                lr = (
                    learning_rate
                    * np.sqrt(1 - beta2 ** (iter + 1))
                    / (1 - beta1 ** (iter + 1))
                )
            else:
                lr = learning_rate

            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            param = x - lr * (v / (np.sqrt(s) + eps))
            return (param, v, s)

        for i in range(train_iters):
            (x, vt, st) = train_one_iter(i, random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-3, atol=1e-3)
    )


def compare_with_numpy_adam_clip_grad(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    betas,
    weight_decay,
    eps,
    do_bias_correction,
    clip_grad_max_norm,
    clip_grad_norm_type,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        adam = flow.optim.Adam(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "clip_grad_max_norm": clip_grad_max_norm,
                    "clip_grad_norm_type": clip_grad_norm_type,
                }
            ],
            do_bias_correction=do_bias_correction,
        )

        def train_one_iter(grad):
            grad_tensor = flow.Tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adam.clip_grad()
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def train_one_iter(iter, grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x

            if do_bias_correction:
                lr = (
                    learning_rate
                    * np.sqrt(1 - beta2 ** (iter + 1))
                    / (1 - beta1 ** (iter + 1))
                )
            else:
                lr = learning_rate

            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            param = x - lr * (v / (np.sqrt(s) + eps))
            return (param, v, s)

        for i in range(train_iters):
            (x, vt, st) = train_one_iter(i, random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-3, atol=1e-3)
    )


@flow.unittest.skip_unless_1n1d()
class TestAdam(flow.unittest.TestCase):
    def test_adam(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9), (0.8, 0.7)]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["eps"] = [1e-08, 1e-07]
        arg_dict["do_bias_correction"] = [True, False]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adam(test_case, *arg)

    def test_adam_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9), (0.8, 0.7)]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["eps"] = [1e-08, 1e-07]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adam_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
