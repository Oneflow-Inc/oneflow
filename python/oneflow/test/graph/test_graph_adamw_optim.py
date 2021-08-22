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


def compare_with_numpy_adamw(
    test_case, device, x_shape, learning_rate, train_iters, betas, weight_decay, eps,
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

    adamw0 = flow.optim.AdamW(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "betas": betas,
                "weight_decay": weight_decay,
            }
        ]
    )

    class CustomAdamWGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adamw0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adamw_graph = CustomAdamWGraph()
    for i in range(train_iters):
        mask_tensor = flow.Tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        adamw_x = adamw_graph(mask_tensor)
        of_res_list.append(simp_module.para0.numpy())

    np_res_list = []

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)

        beta1 = betas[0]
        beta2 = betas[1]

        def np_train_one_iter(grad):
            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            g = (
                learning_rate / (np.sqrt(s) + 1e-08) * v
                + learning_rate * weight_decay * x
            )
            param = x - g
            return (param, v, s)

        for i in range(train_iters):
            (x, vt, st) = np_train_one_iter(random_grad_seq[i])
            np_res_list.append(x)

    train_by_numpy()

    test_case.assertTrue(
        np.allclose(np_res_list, of_res_list, rtol=0.0001, atol=0.0001)
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

    adamw0 = flow.optim.AdamW(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "betas": betas,
                "weight_decay": weight_decay,
                "clip_grad_max_norm": clip_grad_max_norm,
                "clip_grad_norm_type": clip_grad_norm_type,
            }
        ]
    )

    class CustomAdamWGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adamw0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adamw_graph = CustomAdamWGraph()
    for i in range(train_iters):
        mask_tensor = flow.Tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        adamw_x = adamw_graph(mask_tensor)
        of_res_list.append(simp_module.para0.numpy())

    np_res_list = []

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)

        beta1 = betas[0]
        beta2 = betas[1]

        def np_train_one_iter(grad):
            norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            g = (
                learning_rate / (np.sqrt(s) + 1e-08) * v
                + learning_rate * weight_decay * x
            )
            param = x - g
            return (param, v, s)

        for i in range(train_iters):
            (x, vt, st) = np_train_one_iter(random_grad_seq[i])
            np_res_list.append(x)

    train_by_numpy()

    test_case.assertTrue(
        np.allclose(np_res_list, of_res_list, rtol=0.0001, atol=0.0001)
    )


@flow.unittest.skip_unless_1n1d()
class TestAdamW(flow.unittest.TestCase):
    def test_adamw(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [1e-3, 0.0]
        arg_dict["eps"] = [1e-8]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamw(test_case, *arg)

    def test_adamw_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [1e-3, 0.0]
        arg_dict["eps"] = [1e-8]
        arg_dict["clip_grad_max_norm"] = [1.0]
        arg_dict["clip_grad_norm_type"] = [2.0]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamw_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
