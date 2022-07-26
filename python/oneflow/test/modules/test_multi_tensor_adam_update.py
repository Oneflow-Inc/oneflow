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


def compare_with_numpy_adam(
    test_case,
    device,
    x_shape,
    tensor_num,
    betas,
    do_bias_correction,
    learning_rate,
    train_iters,
):
    random_grad_seq = []
    init_value_seq = []
    m_init_value_seq = []
    v_init_value_seq = []

    for _ in range(train_iters):
        random_grad_seq_per_iter = []
        for i in range(tensor_num):
            random_grad_seq_per_iter.append(
                np.random.uniform(size=x_shape).astype(np.float32)
            )
        random_grad_seq.append(random_grad_seq_per_iter)

    for i in range(tensor_num):
        init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
        m_init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
        v_init_value_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    def train_by_oneflow():
        x_tensor_list = []
        m_tensor_list = []
        v_tensor_list = []

        for i in range(tensor_num):
            x_tensor_list.append(
                flow.Tensor(init_value_seq[i], device=flow.device(device))
            )
            m_tensor_list.append(
                flow.Tensor(m_init_value_seq[i], device=flow.device(device))
            )
            v_tensor_list.append(
                flow.Tensor(v_init_value_seq[i], device=flow.device(device))
            )
        lr_tensor = flow.Tensor(learning_rate, device=flow.device(device))
        beta1, beta2 = betas

        def train_one_iter(step, grad):
            bias_correction1 = 1.0
            bias_correction2 = 1.0

            if do_bias_correction:
                bias_correction1 = 1.0 - np.power(beta1, step)
                bias_correction2 = 1.0 - np.power(beta2, step)

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

            flow._C.multi_tensor_adam_update(
                x_tensor_list,
                grad_tensor_list,
                m_tensor_list,
                v_tensor_list,
                lr_tensor,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                do_bias_correction,
                1.0,
                0.0,
            )

        for i in range(1, train_iters + 1):
            train_one_iter(i, random_grad_seq[i - 1])
        return x_tensor_list, m_tensor_list, v_tensor_list

    def train_by_numpy():
        x = init_value_seq
        m = m_init_value_seq
        v = v_init_value_seq
        beta1, beta2 = betas

        def train_one_iter(step, grad):
            for i in range(tensor_num):
                bias_correction1 = 1.0
                bias_correction2 = 1.0

                if do_bias_correction:
                    bias_correction1 = 1.0 - np.power(beta1, step)
                    bias_correction2 = 1.0 - np.power(beta2, step)

                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]
                denom = np.sqrt(v[i]) / np.sqrt(bias_correction2) + 1e-5

                x[i] = x[i] - ((learning_rate / bias_correction1) * m[i] / denom)

            return x

        for i in range(1, train_iters + 1):
            x = train_one_iter(i, random_grad_seq[i - 1])
        return x, m, v

    oneflow_res_list = train_by_oneflow()
    numpy_res_list = train_by_numpy()

    # Test x, m, v
    for tensor_idx in range(3):
        for i in range(tensor_num):
            test_case.assertTrue(
                np.allclose(
                    oneflow_res_list[tensor_idx][i].numpy().flatten(),
                    numpy_res_list[tensor_idx][i].flatten(),
                    rtol=1e-3,
                    atol=1e-3,
                )
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_multi_tensor_adam_update(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["x_shape"] = [(4,)]
        arg_dict["tensor_num"] = [4]
        arg_dict["betas"] = [(0.9, 0.999)]
        arg_dict["do_bias_correction"] = [True, False]
        arg_dict["learning_rate"] = [1.0, 1e-3]
        arg_dict["train_iters"] = [10]

        for arg in GenArgDict(arg_dict):
            compare_with_numpy_adam(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
