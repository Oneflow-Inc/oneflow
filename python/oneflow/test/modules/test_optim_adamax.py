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
import torch as torch_ori
from oneflow.test_utils.test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_adamax(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    betas,
    weight_decay,
    eps,
    do_bias_correction,
    reload_state_step,
    save_load_by_pickle,
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

        adamax = flow.optim.Adamax(
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
            adamax.step()
            adamax.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adamax.state_dict()
                adamax = flow.optim.Adamax([{"params": x,}],)
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir)
                        state_dict = flow.load(save_dir)
                adamax.load_state_dict(state_dict)
        return x

    def train_by_torch():
        x = []
        for i in range(tensor_num):
            x.append(
                torch_ori.nn.Parameter(
                    torch_ori.Tensor(init_value_seq[i]).to(
                        device=torch_ori.device(device)
                    )
                )
            )

        adamax = torch_ori.optim.Adamax(
            [
                {
                    "params": x,
                    "lr": learning_rate,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ],
            # do_bias_correction=do_bias_correction,
            # fused=fused,
        )

        def train_one_iter(grad):
            loss = 0.0
            for i in range(tensor_num):
                grad_tensor = torch_ori.tensor(
                    grad[i],
                    dtype=torch_ori.float32,
                    requires_grad=False,
                    device=torch_ori.device(device),
                )
                loss += torch_ori.sum(x[i] * grad_tensor)
            loss.backward()
            adamax.step()
            adamax.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x

    def train_by_numpy(tensor_idx):
        x = init_value_seq[tensor_idx]
        mt = np.zeros_like(x)
        normt = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def np_train_one_iter(step, grad):
            grad = grad + weight_decay * x
            bias_correction1 = 1.0

            if do_bias_correction:
                bias_correction1 = 1.0 - np.power(beta1, step)

            m = beta1 * mt + (1 - beta1) * grad
            norm = np.maximum(beta2 * normt, abs(grad) + eps)
            param = x - learning_rate * m / (bias_correction1 * norm)
            return (param, m, norm)

        for i in range(1, train_iters + 1):
            (x, mt, normt) = np_train_one_iter(i, random_grad_seq[i - 1][tensor_idx])
        return x

    oneflow_res = train_by_oneflow()
    numpy_res = []
    for i in range(tensor_num):
        numpy_res.append(train_by_numpy(i))
    torch_res = train_by_torch()

    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                oneflow_res[i].numpy().flatten(),
                numpy_res[i].flatten(),
                rtol=0.0001,
                atol=0.0001,
            )
        )


def compare_with_numpy_adamax_clip_grad(
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
    reload_state_step,
    save_load_by_pickle,
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

        adamax = flow.optim.Adamax(
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
            adamax.clip_grad()
            adamax.step()
            adamax.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            if i == reload_state_step:
                state_dict = adamax.state_dict()
                adamax = flow.optim.Adamax([{"params": x,}])
                if save_load_by_pickle:
                    with tempfile.TemporaryDirectory() as save_dir:
                        flow.save(state_dict, save_dir)
                        state_dict = flow.load(save_dir)
                adamax.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value_seq
        mt = np.zeros_like(x)
        normt = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def train_one_iter(step, grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )

            for i in range(tensor_num):
                grad[i] = grad[i] + weight_decay * x[i]

                bias_correction1 = 1.0

                if do_bias_correction:
                    bias_correction1 = 1.0 - np.power(beta1, step)

                mt[i] = beta1 * mt[i] + (1 - beta1) * grad[i]
                normt[i] = np.maximum(beta2 * normt[i], abs(grad[i]) + eps)

                x[i] = x[i] - ((learning_rate / bias_correction1) * mt[i] / normt[i])

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
class TestAdamax(flow.unittest.TestCase):
    def test_adamax(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.9, 0.999)]
        arg_dict["weight_decay"] = [0.9, 0.000]
        arg_dict["eps"] = [1e-08]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        # TODO(WangYi): support fused adamax
        arg_dict["fused"] = [False]
        arg_dict["tensor_num"] = [1, 4]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamax(test_case, *arg)

    def test_adamax_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [0.1, 0.000]
        arg_dict["eps"] = [1e-08]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["clip_grad_max_norm"] = [0, 0.5, 1.0]
        arg_dict["clip_grad_norm_type"] = ["inf", "-inf", 0.0, 1.0, 2.0, 3.5]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [False, True]
        # TODO(WangYi): support fused adamax
        arg_dict["fused"] = [False]
        arg_dict["tensor_num"] = [1, 4]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamax_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
