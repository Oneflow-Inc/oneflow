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

# dynamic memory allocation can't be tested in unittest
os.environ["ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION"] = "0"
import numpy as np
from oneflow.test_utils.test_util import GenArgDict
from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_ftrl(
    test_case,
    weight_decay,
    lr_power,
    lambda1,
    lambda2,
    beta,
    scale,
    learning_rate,
    train_iters,
    use_optional_tensor,
):
    num_rows = 500
    embedding_size = 128
    model_shape = (num_rows, embedding_size)
    line_size = embedding_size * 3

    num_valid_seq = np.random.randint(1, num_rows, (train_iters))
    skip_if_seq = [np.random.randint(2) for i in range(train_iters)]
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=model_shape).astype(np.float32))

    init_value = np.random.uniform(size=(num_rows, line_size)).astype(np.float32)

    down_scale_by = 10

    """
    In OneFlow's optimizer, learning_rate is passed by attr in eager mode, and passed by tensor in lazy mode.
    in this test, if use_optional_tensor is True, we also pass lr_tensor/down_scale_by_tensor/skip_if tensor for unittest.
    if use_optional_tensor is False, we only pass lr by attr, and not have down_scale_by_tensor/skip_if, so mul down_scale_by to scale and skip skip_if's test.
    """
    if use_optional_tensor:
        scale_val = scale
    else:
        # if pass as attr instead of tensor, mul down_scale_by to scale_value
        scale_val = scale / down_scale_by

    class TestGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(
            self,
            ids,
            unique_embeddings,
            embedding_grad,
            lr_tensor,
            down_scale_by_tensor,
            skip_if,
        ):
            # add id shuffle to set num_unique in op, and use it in update
            (_, _, num_valid, _, _, _,) = flow._C.one_embedding_id_shuffle(
                ids, table_ids=None, num_tables=1, embedding_name=""
            )
            return flow._C.one_embedding_ftrl_update(
                num_valid,
                unique_embeddings,
                embedding_grad,
                lr_tensor,
                down_scale_by_tensor,
                skip_if,
                learning_rate,
                scale_val,
                weight_decay,
                lr_power,
                lambda1,
                lambda2,
                beta,
                line_size,
                embedding_size,
                embedding_name="",
            )

    graph = TestGraph()

    def ftrl_by_oneflow():
        unique_embeddings_tensor = flow.tensor(init_value, requires_grad=False).to(
            "cuda"
        )
        if use_optional_tensor:
            lr_tensor = flow.tensor(
                np.array(learning_rate).reshape(1,).astype(np.float32)
            ).to("cuda")
            down_scale_by_tensor = flow.tensor(
                np.array(down_scale_by).reshape(1,).astype(np.float32)
            ).to("cuda")
        else:
            lr_tensor = None
            down_scale_by_tensor = None

        def train_one_iter(ids, unique_embeddings, embedding_grad, skip_if):
            return graph(
                ids,
                unique_embeddings,
                embedding_grad,
                lr_tensor,
                down_scale_by_tensor,
                skip_if,
            )

        for i in range(1, train_iters):
            np_ids = np.zeros(num_rows)
            np_ids[0 : num_valid_seq[i]] = np.arange(num_valid_seq[i])
            # add ids of num_valid unique to use id_shuffle out_put num_unique as grad input
            ids = flow.tensor(np_ids.astype(np.int32)).to("cuda")
            grad_tensor = flow.tensor(random_grad_seq[i]).to("cuda")
            if use_optional_tensor:
                skip_if_tensor = flow.tensor(
                    np.array(skip_if_seq[i]).reshape(1,).astype(np.int64)
                ).to("cuda")
            else:
                skip_if_tensor = None

            updated_tensor = train_one_iter(
                ids, unique_embeddings_tensor, grad_tensor, skip_if_tensor,
            )
            unique_embeddings_tensor[0 : num_valid_seq[i]] = updated_tensor[
                0 : num_valid_seq[i]
            ]
        return unique_embeddings_tensor

    def ftrl_by_numpy():
        x = init_value[:, 0:embedding_size]
        accumulate = init_value[:, embedding_size : 2 * embedding_size]
        z = init_value[:, 2 * embedding_size :]

        def train_one_iter(iter, num_valid, grad, model, accum, z):
            grad[0:num_valid] = grad[0:num_valid] * (scale / down_scale_by)

            new_accum = accumulate[0:num_valid] + grad[0:num_valid] * grad[0:num_valid]

            sigma = (
                np.power(new_accum, lr_power)
                - np.power(accumulate[0:num_valid], lr_power)
            ) / learning_rate

            new_z_val = z[0:num_valid] + grad[0:num_valid] - sigma * model[0:num_valid]

            # Here weight_decay equals to AdamW's, not equal to l2.
            update_val = (np.sign(new_z_val) * lambda1 - new_z_val) / (
                (beta + np.power(new_accum, lr_power)) / learning_rate + lambda2
            ) - learning_rate * weight_decay * model[0:num_valid]

            model[0:num_valid] = np.where(np.abs(new_z_val) < lambda1, 0.0, update_val)
            accumulate[0:num_valid] = new_accum
            z[0:num_valid] = new_z_val

            return (model, accumulate, z)

        for i in range(1, train_iters):
            # when use_optional_tensor is False, not pass skip_if to op
            if skip_if_seq[i] > 0 and use_optional_tensor:
                pass
            else:
                (x, accumulate, z) = train_one_iter(
                    i, int(num_valid_seq[i]), random_grad_seq[i], x, accumulate, z
                )

        return x, accumulate, z

    oneflow_res = ftrl_by_oneflow().numpy()
    of_model = oneflow_res[:, 0:embedding_size]
    of_accum = oneflow_res[:, embedding_size : 2 * embedding_size]
    of_z = oneflow_res[:, 2 * embedding_size :]

    np_model, np_accum, np_z = ftrl_by_numpy()
    test_case.assertTrue(
        np.allclose(of_model.flatten(), np_model.flatten(), rtol=1e-4, atol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(of_accum.flatten(), np_accum.flatten(), rtol=1e-4, atol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(of_z.flatten(), np_z.flatten(), rtol=1e-4, atol=1e-4)
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_ftrl(test_case):
        arg_dict = OrderedDict()
        arg_dict["weight_decay"] = [
            0.0
        ]  # TODO(zzk): Currently Only support weight_decay = 0.0.
        arg_dict["lr_power"] = [-0.2, -0.05]
        arg_dict["lambda1"] = [0.1]
        arg_dict["lambda2"] = [0.00]
        arg_dict["beta"] = [1.0]
        arg_dict["scale"] = [1, 0.1]
        arg_dict["learning_rate"] = [0.3, 1.5]
        arg_dict["train_iters"] = [10]
        arg_dict["use_optional_tensor"] = [True, False]

        for arg in GenArgDict(arg_dict):
            compare_with_numpy_ftrl(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
