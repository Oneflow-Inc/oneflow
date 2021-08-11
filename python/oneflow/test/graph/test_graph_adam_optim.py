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
from test_util import GenArgList
import numpy as np

import oneflow as flow
import oneflow.unittest


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

    adam0 = flow.optim.Adam(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        ],
        do_bias_correction=do_bias_correction,
    )

    class CustomAdamGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer("adam", adam0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    adam_graph = CustomAdamGraph()

    for i in range(train_iters):
        mask_tensor = flow.Tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        adam_x = adam_graph(mask_tensor)

        of_res_list.append(simp_module.para0.numpy())

    np_res_list = []

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def np_train_one_iter(iter, grad):
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
            (x, vt, st) = np_train_one_iter(i, random_grad_seq[i])
            np_res_list.append(x)
        return x

    train_by_numpy()

    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=0.001, atol=0.001))


@flow.unittest.skip_unless_1n1d()
class TestAdam(flow.unittest.TestCase):
    def test_adam(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [0.001, 0.0]
        arg_dict["eps"] = [1e-8]
        arg_dict["do_bias_correction"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adam(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
