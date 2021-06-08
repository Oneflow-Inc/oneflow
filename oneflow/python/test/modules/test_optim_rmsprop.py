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
import oneflow.experimental as flow
from test_util import GenArgList
from oneflow.python.nn.parameter import Parameter


def compare_with_numpy_rmsprop(
    test_case,
    device,
    x_shape,
    scale,
    learning_rate,
    momentum,
    train_iters,
    alpha,
    eps,
    weight_decay,
    centered,
):
    # generate random number sequences
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        param_list = list()
        param_list.append(x)
        rmsprop = flow.optim.RMSprop(
            [
                {
                    "params": param_list,
                    "lr": learning_rate,
                    "alpha": alpha,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "centered": centered,
                    "scale": scale,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.Tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            rmsprop.step()
            rmsprop.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x

    def train_by_numpy():
        x = init_value
        r = np.zeros_like(x)
        v = np.zeros_like(x)
        g = np.zeros_like(x)

        def train_one_iter(grad):
            grad = grad * scale

            if centered:
                r_ = alpha * r + (1 - alpha) * grad * grad
                g_ = alpha * g + (1 - alpha) * grad
                v_ = momentum * v + learning_rate / np.sqrt(r_ - g_ * g_ + eps) * grad
            else:
                r_ = alpha * r + (1 - alpha) * grad * grad
                g_ = g
                v_ = momentum * v + learning_rate / np.sqrt(r_ + eps) * grad

            param = x - v_

            return param, r_, g_, v_

        for i in range(train_iters):
            x, r, g, v = train_one_iter(random_grad_seq[i])

        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=1e-4, atol=1e-4)
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestRMSProp(flow.unittest.TestCase):
    def test_rmsprop(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["scale"] = [1.0, 0.9]
        arg_dict["learning_rate"] = [1]
        arg_dict["momentum"] = [0.0]  # TODO: support nonzero momentum
        arg_dict["train_iters"] = [10]
        arg_dict["alpha"] = [0.9, 0.99]
        arg_dict["eps"] = [1e-8, 1e-5]
        arg_dict["weight_decay"] = [0.1, 0.99]
        arg_dict["centered"] = [False, True]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_rmsprop(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
