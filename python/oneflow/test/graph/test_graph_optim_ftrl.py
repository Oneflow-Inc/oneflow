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
from oneflow.one_embedding import Ftrl


def compare_with_numpy_ftrl(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    weight_decay,
    lr_power,
    initial_accumulator_value,
    lambda1,
    lambda2,
    beta,
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

    ftrl = Ftrl(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "lr_power": lr_power,
                "initial_accumulator_value": initial_accumulator_value,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "beta": beta,
            }
        ]
    )

    class CustomftrlGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(ftrl)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    ftrl_graph = CustomftrlGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        ftrl_x = ftrl_graph(mask_tensor)

        of_res_list.append(copy.copy(simp_module.para0.numpy()))

    np_res_list = []

    def train_by_numpy():
        x = init_value
        accum = np.zeros_like(x)
        accum.fill(initial_accumulator_value)
        z_arr = np.zeros_like(x)

        def np_train_one_iter(grad):
            grad = grad + weight_decay * x

            new_accum = accum + grad * grad
            sigma = (
                np.power(new_accum, lr_power) - np.power(accum, lr_power)
            ) / learning_rate
            new_z_val = z_arr + grad - sigma * x

            update_val = (np.sign(new_z_val) * lambda1 - new_z_val) / (
                (beta + np.power(new_accum, lr_power)) / learning_rate + lambda2
            )
            param = np.where(np.abs(new_z_val) < lambda1, 0.0, update_val)
            return (param, new_accum, new_z_val)

        for i in range(1, train_iters + 1):
            (x, accum, z_arr) = np_train_one_iter(random_grad_seq[i - 1])
            np_res_list.append(x)
        return x

    train_by_numpy()
    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=1e-4, atol=1e-4))


def compare_with_numpy_ftrl_clip_grad(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    weight_decay,
    lr_power,
    initial_accumulator_value,
    lambda1,
    lambda2,
    beta,
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

    ftrl = Ftrl(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "lr_power": lr_power,
                "initial_accumulator_value": initial_accumulator_value,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "beta": beta,
                "clip_grad_max_norm": clip_grad_max_norm,
                "clip_grad_norm_type": clip_grad_norm_type,
            }
        ]
    )

    class CustomftrlGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(ftrl)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    ftrl_graph = CustomftrlGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        ftrl_x = ftrl_graph(mask_tensor)

        of_res_list.append(copy.copy(simp_module.para0.numpy()))

    np_res_list = []

    def train_by_numpy():
        x = init_value
        accum = np.zeros_like(x)
        accum.fill(initial_accumulator_value)
        z_arr = np.zeros_like(x)

        def np_train_one_iter(grad):
            norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x

            new_accum = accum + grad * grad
            sigma = (
                np.power(new_accum, lr_power) - np.power(accum, lr_power)
            ) / learning_rate
            new_z_val = z_arr + grad - sigma * x

            update_val = (np.sign(new_z_val) * lambda1 - new_z_val) / (
                (beta + np.power(new_accum, lr_power)) / learning_rate + lambda2
            )
            param = np.where(np.abs(new_z_val) < lambda1, 0.0, update_val)
            return (param, new_accum, new_z_val)

        for i in range(1, train_iters + 1):
            (x, accum, z_arr) = np_train_one_iter(random_grad_seq[i - 1])
            np_res_list.append(x)
        return x

    train_by_numpy()

    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=1e-4, atol=1e-4))


@flow.unittest.skip_unless_1n1d()
class Testftrl(flow.unittest.TestCase):
    def test_ftrl(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["weight_decay"] = [0.9, 0.000]
        arg_dict["lr_power"] = [-0.5, 0.5]
        arg_dict["initial_accumulator_value"] = [0.1, 0.05]
        arg_dict["lambda1"] = [0.01]
        arg_dict["lambda2"] = [0.0, 0.01]
        arg_dict["beta"] = [1.0]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_ftrl(test_case, *arg)

    def test_ftrl_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["weight_decay"] = [0.9, 0.000]
        arg_dict["lr_power"] = [-0.5, 0.5]
        arg_dict["initial_accumulator_value"] = [0.1, 0.05]
        arg_dict["lambda1"] = [0.01]
        arg_dict["lambda2"] = [0.0, 0.01]
        arg_dict["beta"] = [1.0]
        arg_dict["clip_grad_max_norm"] = [1.0]
        arg_dict["clip_grad_norm_type"] = [2.0]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_ftrl_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
