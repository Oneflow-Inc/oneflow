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
import os
from collections import OrderedDict
import numpy as np
from test_util import GenArgList
import oneflow as flow
import oneflow.unittest


def compare_with_numpy_sgd(
    test_case, device, x_shape, learning_rate, train_iters, momentum, weight_decay
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

    sgd0 = flow.optim.SGD(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "momentum": momentum,
                "weight_decay": weight_decay,
            }
        ],
    )

    class CustomSGDGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer("sgd", sgd0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    sgd_graph = CustomSGDGraph()
    for i in range(train_iters):
        mask_tensor = flow.Tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        sgd_x = sgd_graph(mask_tensor)
        of_res_list.append(simp_module.para0.numpy())

    np_res_list = []

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)

        def np_train_one_iter(grad):
            grad = grad + weight_decay * x
            v = momentum * vt - learning_rate * grad
            param = x + v
            return (param, v)

        for i in range(train_iters):
            (x, vt) = np_train_one_iter(random_grad_seq[i])
            np_res_list.append(x)

    train_by_numpy()
    test_case.assertTrue(np.allclose(np_res_list, of_res_list, rtol=0.001, atol=0.001))


@flow.unittest.skip_unless_1n1d()
class TestCpuSGD(flow.unittest.TestCase):
    def test_sgd(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["momentum"] = [0.9, 0.8]
        arg_dict["weight_decay"] = [0.001, 0.0]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_sgd(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
