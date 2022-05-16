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
import unittest
from collections import OrderedDict
import numpy as np
import copy

from test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow


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
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.para0 = flow.nn.Parameter(
                flow.Tensor(init_value, device=flow.device(device))
            )

        def forward(self, mask):
            return self.para0 * mask

    simp_module = CustomModule()
    simp_module.to(device)
    simp_module.train()

    adam0 = flow.optim.Adagrad(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        ],
        lr_decay=lr_decay,
        initial_accumulator_value=initial_accumulator_value,
    )

    class CustomAdagradGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adam0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adagrad_graph = CustomAdagradGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        adagrad_x = adagrad_graph(mask_tensor)

        of_res_list.append(copy.copy(simp_module.para0.numpy()))

    np_res_list = []

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
            np_res_list.append(x)
        return x

    train_by_numpy()
    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=0.001, atol=0.001))


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
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.para0 = flow.nn.Parameter(
                flow.Tensor(init_value, device=flow.device(device))
            )

        def forward(self, mask):
            return self.para0 * mask

    simp_module = CustomModule()
    simp_module.to(device)
    simp_module.train()

    adam0 = flow.optim.Adagrad(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "eps": eps,
                "weight_decay": weight_decay,
                "clip_grad_max_norm": clip_grad_max_norm,
                "clip_grad_norm_type": clip_grad_norm_type,
            }
        ],
        lr_decay=lr_decay,
        initial_accumulator_value=initial_accumulator_value,
    )

    class CustomAdagradGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adam0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adagrad_graph = CustomAdagradGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        adagrad_x = adagrad_graph(mask_tensor)

        of_res_list.append(copy.copy(simp_module.para0.numpy()))

    np_res_list = []

    def train_by_numpy():
        x = init_value
        st = np.ones_like(x) * initial_accumulator_value

        def np_train_one_iter(iter, grad):
            norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x
            lr = learning_rate / (1 + (iter - 1) * lr_decay)
            s = st + grad * grad
            param = x - lr / (np.sqrt(s) + eps) * grad

            return (param, s)

        for i in range(1, train_iters + 1):
            (x, st) = np_train_one_iter(i, random_grad_seq[i - 1])
            np_res_list.append(x)

        return x

    train_by_numpy()

    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=0.001, atol=0.001))


@flow.unittest.skip_unless_1n1d()
class TestAdagrad(flow.unittest.TestCase):
    def test_adagrad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["lr_decay"] = [0.9, 0.75]
        arg_dict["weight_decay"] = [0.0, 0.1]
        arg_dict["initial_accumulator_value"] = [1.0, 2.1]
        arg_dict["eps"] = [1e-08, 1e-07]

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
        arg_dict["weight_decay"] = [0.0, 0.9]
        arg_dict["initial_accumulator_value"] = [1.0, 2.1]
        arg_dict["eps"] = [1e-8]
        arg_dict["clip_grad_max_norm"] = [1.0]
        arg_dict["clip_grad_norm_type"] = [2.0]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adagrad_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
