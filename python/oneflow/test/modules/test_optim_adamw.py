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
import random as random_util

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import random_bool, random_device
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_adamw(
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
    contiguous_params,
    fused,
    tensor_num,
):
    random_grad_seq = []
    init_value_seq = []

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    def train_by_oneflow():
        x = []
        for i in range(tensor_num):
            x.append(
                Parameter(flow.Tensor(init_value_seq[i], device=flow.device(device)))
            )

        adam = flow.optim.AdamW(
            [
                {
                    "params": x,
                    "lr": learning_rate,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ],
            do_bias_correction=do_bias_correction,
            amsgrad=amsgrad,
            contiguous_params=contiguous_params,
            fused=fused,
        )

        def train_one_iter(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = flow.tensor(
                    grad[i],
                    dtype=flow.float32,
                    requires_grad=False,
                    device=flow.device(device),
                )
                loss += flow.sum(x[i] * grad_tensor)
            loss.backward()
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adam.state_dict()
                adam = flow.optim.AdamW(x, contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                adam.load_state_dict(state_dict)
        return x

    def train_by_numpy(tensor_idx):
        x = init_value_seq[tensor_idx]
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        max_st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def train_one_iter(step, grad):
            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad

            bias_correction1 = 1.0
            bias_correction2 = 1.0

            if do_bias_correction:
                bias_correction1 = 1.0 - np.power(beta1, step)
                bias_correction2 = 1.0 - np.power(beta2, step)

            max_s = np.zeros_like(x)
            if amsgrad:
                max_s = np.maximum(s, max_st)
                denom = np.sqrt(max_s) / np.sqrt(bias_correction2) + eps
            else:
                denom = np.sqrt(s) / np.sqrt(bias_correction2) + eps

            lr = learning_rate / bias_correction1 / denom
            g = lr * v + learning_rate * weight_decay * x
            param = x - g
            return (param, v, s, max_s)

        for i in range(1, train_iters + 1):
            (x, vt, st, max_st) = train_one_iter(i, random_grad_seq[i - 1][tensor_idx])
        return x

    oneflow_res = train_by_oneflow()
    numpy_res = []
    for i in range(tensor_num):
        numpy_res.append(train_by_numpy(i))

    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                oneflow_res[i].numpy().flatten(),
                numpy_res[i].flatten(),
                rtol=0.0001,
                atol=0.0001,
            )
        )


def compare_with_numpy_adamw_clip_grad(
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
    contiguous_params,
    fused,
    tensor_num,
):
    random_grad_seq = []
    init_value_seq = []

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    def train_by_oneflow():
        x = []
        for i in range(tensor_num):
            x.append(
                Parameter(flow.Tensor(init_value_seq[i], device=flow.device(device)))
            )

        adam = flow.optim.AdamW(
            [
                {
                    "params": x,
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
            contiguous_params=contiguous_params,
            fused=fused,
        )

        def train_one_iter(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = flow.tensor(
                    grad[i],
                    dtype=flow.float32,
                    requires_grad=False,
                    device=flow.device(device),
                )
                loss += flow.sum(x[i] * grad_tensor)
            loss.backward()
            adam.clip_grad()
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adam.state_dict()
                adam = flow.optim.AdamW(x, contiguous_params=contiguous_params)
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile() as f:
                        flow.save(state_dict, f.name)
                        state_dict = flow.load(f.name)
                adam.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value_seq
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        max_st = np.zeros_like(x)

        beta1 = betas[0]
        beta2 = betas[1]

        def train_one_iter(step, grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )

            for i in range(tensor_num):
                vt[i] = beta1 * vt[i] + (1 - beta1) * grad[i]
                st[i] = beta2 * st[i] + (1 - beta2) * grad[i] * grad[i]

                bias_correction1 = 1.0
                bias_correction2 = 1.0

                if do_bias_correction:
                    bias_correction1 = 1.0 - np.power(beta1, step)
                    bias_correction2 = 1.0 - np.power(beta2, step)

                if amsgrad:
                    max_st[i] = np.maximum(st[i], max_st[i])
                    denom = np.sqrt(max_st[i]) / np.sqrt(bias_correction2) + eps
                else:
                    denom = np.sqrt(st[i]) / np.sqrt(bias_correction2) + eps

                lr = learning_rate / bias_correction1 / denom
                g = lr * vt[i] + learning_rate * weight_decay * x[i]
                x[i] = x[i] - g

        for i in range(1, train_iters + 1):
            train_one_iter(i, random_grad_seq[i - 1])
        return x

    oneflow_res = train_by_oneflow()
    numpy_res = train_by_numpy()

    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                oneflow_res[i].numpy().flatten(),
                numpy_res[i].flatten(),
                rtol=0.0001,
                atol=0.0001,
            )
        )


@flow.unittest.skip_unless_1n1d()
class TestAdamW(flow.unittest.TestCase):
    def test_adamw(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = [random_device().value()]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.9, 0.999)]
        arg_dict["weight_decay"] = [0.01, 0.00]
        arg_dict["eps"] = [1e-8]
        arg_dict["do_bias_correction"] = [random_bool().value()]
        arg_dict["amsgrad"] = [random_bool().value()]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [random_bool().value()]
        arg_dict["contiguous_params"] = [random_bool().value()]
        arg_dict["fused"] = [random_bool().value()]
        arg_dict["tensor_num"] = [1, 4]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamw(test_case, *arg)

    def test_adamw_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = [random_device().value()]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.9, 0.999)]
        arg_dict["weight_decay"] = [0.001, 0.0]
        arg_dict["eps"] = [1e-8]
        arg_dict["do_bias_correction"] = [random_bool().value()]
        arg_dict["amsgrad"] = [random_bool().value()]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = random_util.sample(
            ["inf", "-inf", 0.0, 1.0, 2.0, 3.5], k=3
        )
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [random_bool().value()]
        arg_dict["contiguous_params"] = [random_bool().value()]
        arg_dict["fused"] = [random_bool().value()]
        arg_dict["tensor_num"] = [1, 4]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamw_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
