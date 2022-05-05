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
import tempfile

import os
import numpy as np
from oneflow.test_utils.test_util import GenArgDict
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_sgd(
    test_case, momentum, weight_decay, scale, learning_rate, train_iters,
):

    num_rows = 500
    embedding_size = 128
    model_shape = (num_rows, embedding_size)
    line_size = embedding_size * 2 if momentum > 0 else embedding_size

    num_valid_seq = np.random.randint(1, num_rows, (train_iters))
    skip_if_seq = [np.random.randint(2) for i in range(train_iters)]

    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=model_shape).astype(np.float32))

    init_value = np.random.uniform(size=(num_rows, line_size)).astype(np.float32)

    down_scale_by = 10

    def sgd_by_oneflow():
        unique_embeddings_tensor = flow.tensor(init_value, requires_grad=False).to(
            "cuda"
        )
        lr_tensor = flow.tensor(
            np.array(learning_rate).reshape(1,).astype(np.float32)
        ).to("cuda")
        down_scale_by_tensor = flow.tensor(
            np.array(down_scale_by).astype(np.float32)
        ).to("cuda")

        def train_one_iter(num_valid, unique_embeddings, embedding_grad, skip_if):
            return flow._C.one_embedding_sgd_update(
                num_valid,
                unique_embeddings,
                embedding_grad,
                lr_tensor,
                down_scale_by_tensor,
                skip_if,
                scale,
                weight_decay,
                momentum,
            )

        for i in range(train_iters):
            num_valid_tensor = flow.tensor(
                np.array(num_valid_seq[i]).reshape(1,).astype(np.int32)
            ).to("cuda")
            grad_tensor = flow.tensor(random_grad_seq[i]).to("cuda")
            skip_if_tensor = flow.tensor(
                np.array(skip_if_seq[i]).reshape(1,).astype(np.int64)
            ).to("cuda")
            updated_tensor = train_one_iter(
                num_valid_tensor, unique_embeddings_tensor, grad_tensor, skip_if_tensor
            )
            unique_embeddings_tensor[0 : num_valid_seq[i]] = updated_tensor[
                0 : num_valid_seq[i]
            ]
        return unique_embeddings_tensor

    def sgd_by_numpy():
        x = init_value[:, 0:embedding_size]
        vt = init_value[:, embedding_size:]

        def train_one_iter(num_valid, grad, model, state):
            grad[0:num_valid] = grad[0:num_valid] * (scale / down_scale_by)
            next_state = (
                momentum * state[0:num_valid] if momentum > 0 else 0
            ) - learning_rate * grad[0:num_valid]
            if momentum > 0:
                state[0:num_valid] = next_state
            model[0:num_valid] = (
                model[0:num_valid]
                + next_state
                - learning_rate * weight_decay * model[0:num_valid]
            )
            return (model, state)

        for i in range(train_iters):
            if skip_if_seq[i] > 0:
                pass
            else:
                (x, vt) = train_one_iter(
                    int(num_valid_seq[i]), random_grad_seq[i], x, vt
                )
        return x, vt

    oneflow_res = sgd_by_oneflow().numpy()
    of_model = oneflow_res[:, 0:embedding_size]
    of_momentum = oneflow_res[:, embedding_size:]
    np_model, np_momentum = sgd_by_numpy()
    test_case.assertTrue(
        np.allclose(of_model.flatten(), np_model.flatten(), rtol=0.001, atol=0.001)
    )
    if momentum > 0:
        test_case.assertTrue(
            np.allclose(
                of_momentum.flatten(), np_momentum.flatten(), rtol=0.001, atol=0.001
            )
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_one_embedding_sgd(test_case):
        arg_dict = OrderedDict()
        arg_dict["momentum"] = [0, 0.9]
        arg_dict["weight_decay"] = [0, 0.1]
        arg_dict["scale"] = [1, 0.1]
        arg_dict["learning_rate"] = [1, 0.9]
        arg_dict["train_iters"] = [10]
        for arg in GenArgDict(arg_dict):
            compare_with_numpy_sgd(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
