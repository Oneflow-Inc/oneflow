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
    amsgrad,
    reload_state_step,
    save_load_by_pickle,
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
            amsgrad=amsgrad,
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
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adam.state_dict()
                adam = flow.optim.Adam([{"params": [x],}],)
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir)
                        state_dict = flow.load(save_dir)
                adam.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        max_st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def np_train_one_iter(step, grad):
            grad = grad + weight_decay * x

            bias_correction1 = 1.0
            bias_correction2 = 1.0

            if do_bias_correction:
                bias_correction1 = 1.0 - np.power(beta1, step)
                bias_correction2 = 1.0 - np.power(beta2, step)

            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            max_s = np.zeros_like(x)

            if amsgrad:
                max_s = np.maximum(s, max_st)
                denom = np.sqrt(max_s) / np.sqrt(bias_correction2) + eps
            else:
                denom = np.sqrt(s) / np.sqrt(bias_correction2) + eps

            param = x - ((learning_rate / bias_correction1) * v / denom)
            return (param, v, s, max_s)

        for i in range(1, train_iters + 1):
            (x, vt, st, max_st) = np_train_one_iter(i, random_grad_seq[i - 1])
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
    amsgrad,
    clip_grad_max_norm,
    clip_grad_norm_type,
    reload_state_step,
    save_load_by_pickle,
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
            amsgrad=amsgrad,
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
            adam.clip_grad()
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adam.state_dict()
                adam = flow.optim.Adam([{"params": [x],}])
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir)
                        state_dict = flow.load(save_dir)
                adam.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        max_st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def train_one_iter(step, grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x

            bias_correction1 = 1.0
            bias_correction2 = 1.0

            if do_bias_correction:
                bias_correction1 = 1.0 - np.power(beta1, step)
                bias_correction2 = 1.0 - np.power(beta2, step)

            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            max_s = np.zeros_like(x)

            if amsgrad:
                max_s = np.maximum(s, max_st)
                denom = np.sqrt(max_s) / np.sqrt(bias_correction2) + eps
            else:
                denom = np.sqrt(s) / np.sqrt(bias_correction2) + eps

            param = x - ((learning_rate / bias_correction1) * v / denom)
            return (param, v, s, max_s)

        for i in range(1, train_iters + 1):
            (x, vt, st, max_st) = train_one_iter(i, random_grad_seq[i - 1])
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
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [0.9, 0.000]
        arg_dict["eps"] = [1e-08]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["amsgrad"] = [True, False]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adam(test_case, *arg)

    def test_adam_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [0.1, 0.000]
        arg_dict["eps"] = [1e-08]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["amsgrad"] = [True, False]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adam_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
