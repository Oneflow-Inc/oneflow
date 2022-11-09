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

from test_util import GenArgList
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow


def compare_with_numpy_lamb(
    test_case,
    device,
    x_shape,
    learning_rate,
    train_iters,
    betas,
    weight_decay,
    eps,
    do_bias_correction,
    adam_w_mode,
    clip_grad_max_norm,
    clip_grad_norm_type,
):

    np.random.seed(1000)

    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = flow.nn.Parameter(
                flow.Tensor(init_value, device=flow.device(device))
            )

        def forward(self, mask):
            return self.param * mask

    simp_module = CustomModule()
    simp_module.to(device)
    simp_module.train()

    optim_kwargs = {
        "params": simp_module.parameters(),
        "lr": learning_rate,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "adam_w_mode": adam_w_mode,
        "do_bias_correction": do_bias_correction,
    }

    if clip_grad_max_norm != -1:
        optim_kwargs["clip_grad_max_norm"] = clip_grad_max_norm
        optim_kwargs["clip_grad_norm_type"] = clip_grad_norm_type

    lamb_optim = flow.optim.LAMB([optim_kwargs])

    class CustomLambGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(lamb_optim)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            return loss

    lamb_graph = CustomLambGraph()

    for i in range(train_iters):
        mask_tensor = flow.tensor(
            random_grad_seq[i],
            dtype=flow.float32,
            requires_grad=False,
            device=flow.device(device),
        )
        lamb_graph(mask_tensor)

    of_res = simp_module.param.numpy()

    def train_by_numpy():
        x = init_value
        mt = np.zeros_like(x)
        vt = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]
        if adam_w_mode:
            l2 = 0
            wd = weight_decay
        else:
            l2 = weight_decay
            wd = 0

        def np_train_one_iter(step, grad):
            if clip_grad_max_norm != -1:
                _, grad = clip_grad_norm_np(
                    grad, clip_grad_max_norm, clip_grad_norm_type
                )

            grad = grad + l2 * x

            bias_correction1 = 1.0
            bias_correction2 = 1.0

            if do_bias_correction:
                bias_correction1 = 1.0 - np.power(beta1, step + 1)
                bias_correction2 = 1.0 - np.power(beta2, step + 1)

            m = beta1 * mt + (1 - beta1) * grad
            v = beta2 * vt + (1 - beta2) * grad * grad

            denom = np.sqrt(v) / np.sqrt(bias_correction2) + eps

            adam_diff = m / bias_correction1 / denom

            w_norm = np.linalg.norm(x, ord=2)
            g_norm = np.linalg.norm(adam_diff, ord=2)
            if w_norm > 0 and g_norm > 0:
                trust_ratio = w_norm / g_norm
            else:
                trust_ratio = 1.0

            param = x - learning_rate * trust_ratio * (adam_diff + wd * x)
            return (param, m, v)

        for i in range(train_iters):
            (x, mt, vt) = np_train_one_iter(i, random_grad_seq[i])
        return x

    np_res = train_by_numpy()

    test_case.assertTrue(
        np.allclose(of_res.flatten(), np_res.flatten(), rtol=1e-3, atol=1e-3)
    )


@flow.unittest.skip_unless_1n1d()
class TestLamb(flow.unittest.TestCase):
    def test_lamb(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["learning_rate"] = [0.1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [0.001, 0.1]
        arg_dict["eps"] = [1e-8, 1e-6]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["adam_w_mode"] = [True, False]
        # NOTE(l1aoxingyu): max_norm = -1 means no clip grad
        # nn.Graph only support `clip_grad_max_norm == 1.0` and `clip_grad_norm_type == 2.0`
        arg_dict["clip_grad_max_norm"] = [-1, 1.0]
        arg_dict["clip_grad_norm_type"] = [2.0]

        for arg in GenArgList(arg_dict):
            compare_with_numpy_lamb(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
