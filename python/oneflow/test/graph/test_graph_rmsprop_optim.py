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
import oneflow as flow
import oneflow.unittest
from test_util import GenArgList


@flow.unittest.skip_unless_1n1d()
def compare_with_numpy_rmsprop(
    test_case,
    device,
    x_shape,
    learning_rate,
    momentum,
    train_iters,
    alpha,
    eps,
    weight_decay,
    centered,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.param0 = flow.nn.Parameter(
                flow.Tensor(init_value, device=flow.device(device))
            )

        def forward(self, mask):
            return self.param0 * mask

    simp_module = CustomModel()
    simp_module.to(flow.device(device))
    simp_module.train()

    rmsprop0 = flow.optim.RMSprop(
        [
            {
                "params": simp_module.parameters(),
                "lr": learning_rate,
                "alpha": alpha,
                "eps": eps,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "centered": centered,
            }
        ]
    )

    class CustomRMSpropGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer("rmsprop", rmsprop0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    of_res_list = []
    rmsprop_graph = CustomRMSpropGraph()

    for i in range(train_iters):
        mask_tensor = flow.Tensor(
            random_grad_seq[i], requires_grad=False, device=flow.device(device)
        )
        rmsprop_x = rmsprop_graph(mask_tensor)

        of_res_list.append(simp_module.param0.numpy())

    np_res_list = []

    def train_by_numpy():
        x = init_value
        r = np.zeros_like(x)
        v = np.zeros_like(x)
        g = np.zeros_like(x)

        def np_train_one_iter(grad):
            # ref to: https://github.com/Oneflow-Inc/oneflow/blob/master/python/oneflow/test/modules/test_optim_rmsprop.py#L78-L99

            # weight decay is equivalent to l2 penalty
            grad = grad + weight_decay * x
            r_ = alpha * r + (1 - alpha) * grad * grad
            if centered:
                g_ = alpha * g + (1 - alpha) * grad
                v_ = momentum * v + learning_rate / np.sqrt(r_ - g_ * g_ + eps) * grad
            else:
                g_ = g
                v_ = momentum * v + learning_rate / np.sqrt(r_ + eps) * grad
            param = x - v_
            return (param, r_, g_, v_)

        for i in range(train_iters):
            (x, r, g, v) = np_train_one_iter(random_grad_seq[i])
            np_res_list.append(x)
        return x

    train_by_numpy()

    test_case.assertTrue(np.allclose(of_res_list, np_res_list, rtol=1e-3, atol=1e-3))


@flow.unittest.skip_unless_1n1d()
class TestRMSprop(flow.unittest.TestCase):
    def test_rmsprop(test_case):
        args_dict = OrderedDict()
        args_dict["device"] = ["cpu", "cuda"]
        args_dict["x_shape"] = [(1,), (10,)]
        args_dict["learning_rate"] = [1, 10]
        args_dict["momentum"] = [0.0]  # not supported momentum > 0
        args_dict["train_iters"] = [10]
        args_dict["alpha"] = [0.9, 0.99]
        args_dict["eps"] = [1e-8, 1e-5]
        args_dict["weight_decay"] = [0.1, 0.9]
        args_dict["centered"] = [False, True]

        for args in GenArgList(args_dict):
            compare_with_numpy_rmsprop(test_case, *args)


if __name__ == "__main__":
    unittest.main()
