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
import os

from test_util import GenArgList

import oneflow as flow


def compare_with_numpy_adam(
    test_case,
    device,
    x_shape,
    tensor_num,
    learning_rate,
    train_iters,
    betas,
    weight_decay,
    eps,
    do_bias_correction,
    amsgrad,
):
    random_weight_seq = []
    init_value_seq = []

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_weight_seq.append(random_grad_seq_per_iter)

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.add_parameters()

        def add_parameters(self) -> None:
            for idx in range(tensor_num):
                self.register_parameter(
                    f"param_{idx}",
                    flow.nn.Parameter(
                        flow.tensor(init_value_seq[idx], device=flow.device(device))
                    ),
                )

        def param(self, i):
            return getattr(self, f"param_{i}")

        def forward(self, mask_list):
            out = 0
            for idx in range(tensor_num):
                out += flow._C.matmul(self.param(idx), mask_list[idx])

            return out

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
            },
        ],
        do_bias_correction=do_bias_correction,
        amsgrad=amsgrad,
    )

    class CustomAdamGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer(adam0)
            self.config.enable_amp(True)
            self.config.allow_fuse_model_update_ops(True)
            self.config.enable_multi_tensor_update(True)
            self.config.enable_fused_model_update_cast(True)

        def build(self, mask_tensor_list):
            loss = flow.sum(self.m(mask_tensor_list))
            loss.backward()
            return loss

    of_res_list = []
    adam_graph = CustomAdamGraph()
    for i in range(train_iters):
        mask_tensor_list = []
        for idx in range(tensor_num):
            mask_tensor_list.append(
                flow.tensor(
                    random_weight_seq[i][idx],
                    dtype=flow.float32,
                    requires_grad=False,
                    device=flow.device(device),
                )
            )
        adam_x = adam_graph(mask_tensor_list)
        of_res_list.append([])
        for idx in range(tensor_num):
            of_res_list[i].append(copy.copy(simp_module.param(idx).numpy()))

    np_res_list = []

    def train_by_numpy():
        x = init_value_seq
        m = []
        v = []
        for idx in range(tensor_num):
            m.append(np.zeros_like(x[idx]))
            v.append(np.zeros_like(x[idx]))
        beta1 = betas[0]
        beta2 = betas[1]

        ones = np.ones(x_shape).astype(np.float32)

        def train_one_iter(step, weight):
            for i in range(tensor_num):
                transposed_weight = np.transpose(weight[i], (1, 0))
                grad = np.matmul(ones, transposed_weight)
                grad = grad + weight_decay * x[i]

                bias_correction1 = 1.0
                bias_correction2 = 1.0

                if do_bias_correction:
                    bias_correction1 = 1.0 - np.power(beta1, step)
                    bias_correction2 = 1.0 - np.power(beta2, step)

                m[i] = beta1 * m[i] + (1 - beta1) * grad
                v[i] = beta2 * v[i] + (1 - beta2) * grad * grad
                denom = np.sqrt(v[i]) / np.sqrt(bias_correction2) + eps

                x[i] = x[i] - ((learning_rate / bias_correction1) * m[i] / denom)
            return (x, m, v)

        for i in range(1, train_iters + 1):
            x, m, v = train_one_iter(i, random_weight_seq[i - 1])
            np_res_list.append(copy.copy(x))

    train_by_numpy()
    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(np_res_list[i], of_res_list[i], rtol=1e-3, atol=1e-3)
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestMultiTensorAdam(flow.unittest.TestCase):
    def test_multi_tensor_adam(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["x_shape"] = [(4, 4)]
        arg_dict["tensor_num"] = [4, 6]
        arg_dict["learning_rate"] = [1, 1e-3]
        arg_dict["train_iters"] = [10]
        arg_dict["betas"] = [(0.99, 0.9)]
        arg_dict["weight_decay"] = [0.0, 1e-3]
        arg_dict["eps"] = [1e-5]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["amsgrad"] = [False]  # Multi tensor update do not support amsgrad
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adam(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
