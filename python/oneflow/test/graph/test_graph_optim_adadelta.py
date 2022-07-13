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
import copy

from test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow


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

    adadelta0 = flow.optim.Adadelta(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ],
        rho=rho,
        eps=eps,
        maximize=maximize,
    )

    class CustomAdadeltaGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adadelta0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adadelta_graph = CustomAdadeltaGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i],
            dtype=flow.float32,
            requires_grad=False,
            device=flow.device(device),
        )
        adadelta_x = adadelta_graph(mask_tensor)

        of_res_list.append(copy.copy(simp_module.para0.numpy()))

    np_res_list = []

    def train_by_numpy():
        x = init_value
        square_avgs = np.zeros_like(x)
        acc_deltas = np.zeros_like(x)

        def np_train_one_iter(grad):
            grad = grad if not maximize else -grad
            grad = grad + weight_decay * x
            new_square_avgs = square_avgs * rho + (1.0 - rho) * grad * grad
            std = np.sqrt(new_square_avgs + eps)
            delta = np.sqrt(acc_deltas + eps) / std * grad
            new_acc_deltas = acc_deltas * rho + delta * delta * (1 - rho)
            param = x - learning_rate * delta
            return (param, new_square_avgs, new_acc_deltas)

        for i in range(1, train_iters + 1):
            (x, square_avgs, acc_deltas) = np_train_one_iter(random_grad_seq[i - 1])
            np_res_list.append(x)
        return x

    train_by_numpy()

    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=1e-4, atol=1e-4))


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
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.para0 = flow.nn.Parameter(
                flow.tensor(init_value, device=flow.device(device))
            )

        def forward(self, mask):
            return self.para0 * mask

    simp_module = CustomModule()
    simp_module.to(device)
    simp_module.train()

    adadelta0 = flow.optim.Adadelta(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "clip_grad_max_norm": clip_grad_max_norm,
                "clip_grad_norm_type": clip_grad_norm_type,
            }
        ],
        rho=rho,
        eps=eps,
        maximize=maximize,
    )

    class CustomAdadeltaGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adadelta0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adadelta_graph = CustomAdadeltaGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        adadelta_x = adadelta_graph(mask_tensor)

        of_res_list.append(copy.copy(simp_module.para0.numpy()))

    np_res_list = []

    def train_by_numpy():
        x = init_value
        square_avgs = np.zeros_like(x)
        acc_deltas = np.zeros_like(x)

        def np_train_one_iter(grad):
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
            (x, square_avgs, acc_deltas) = np_train_one_iter(random_grad_seq[i - 1])
            np_res_list.append(x)
        return x

    train_by_numpy()
    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=1e-4, atol=1e-4))


@flow.unittest.skip_unless_1n1d()
class TestAdadelta(flow.unittest.TestCase):
    def test_adadelta(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["rho"] = [0.9]
        arg_dict["eps"] = [1e-6]
        arg_dict["maximize"] = [False]
        arg_dict["weight_decay"] = [0.1]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_adadelta(test_case, *arg)

    def test_adadelta_clip_grad(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["rho"] = [0.9]
        arg_dict["eps"] = [1e-6]
        arg_dict["maximize"] = [False]
        arg_dict["weight_decay"] = [0.1]
        arg_dict["clip_grad_max_norm"] = [1.0]
        arg_dict["clip_grad_norm_type"] = [2.0]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adadelta_clip_grad(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
