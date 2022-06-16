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
import os
import numpy as np
from oneflow.test_utils.test_util import GenArgDict

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_sgd(
    test_case, device, x_shape, tensor_num, weight_decay, learning_rate, train_iters
):
    random_grad_seq = []
    init_value_seq = []

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    def train_by_oneflow():
        x_tensor_list = []
        for i in range(tensor_num):
            x_tensor_list.append(
                flow.Tensor(init_value_seq[i], device=flow.device(device))
            )
        lr_tensor = flow.Tensor(learning_rate, device=flow.device(device))

        def train_one_iter(grad):
            grad_tensor_list = []
            for i in range(tensor_num):
                grad_tensor_list.append(
                    flow.tensor(
                        grad[i],
                        dtype=flow.float32,
                        requires_grad=False,
                        device=flow.device(device),
                    )
                )

            flow._C.multi_tensor_sgd_update(
                x_tensor_list, grad_tensor_list, lr_tensor, 1.0, weight_decay
            )

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x_tensor_list

    def train_by_numpy():
        x = init_value_seq

        def train_one_iter(grad):
            for i in range(tensor_num):
                grad[i] = grad[i] + weight_decay * x[i]
                x[i] = x[i] - learning_rate * grad[i]
            return x

        for i in range(train_iters):
            x = train_one_iter(random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow()
    numpy_res = train_by_numpy()

    for i in range(tensor_num):
        test_case.assertTrue(
            np.allclose(
                oneflow_res[i].numpy().flatten(),
                numpy_res[i].flatten(),
                rtol=0.0001,
                atol=0.0001,
            )
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_multi_tensor_sgd_update(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["x_shape"] = [(2,)]
        arg_dict["tensor_num"] = [4]
        arg_dict["weight_decay"] = [0.0, 0.5]
        arg_dict["learning_rate"] = [1.0, 1e-3]
        arg_dict["train_iters"] = [10]
        for arg in GenArgDict(arg_dict):
            compare_with_numpy_sgd(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
