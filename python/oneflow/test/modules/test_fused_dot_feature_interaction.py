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
from oneflow.test_utils.test_util import GenArgDict
import numpy as np
import oneflow as flow
import oneflow.unittest
import os


def _test_fused_dot_feature_interaction(
    test_case,
    embedding_size,
    self_interaction=False,
    output_concat=True,
    output_padding=0,
    dtype=flow.float32,
    device_type="cuda",
):
    batch_size = 100
    dims = 26
    if dtype == flow.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    feature_0_np = np.random.rand(batch_size, embedding_size).astype(np_dtype)
    feature_1_np = np.random.rand(batch_size, 26, embedding_size).astype(np_dtype)
    feature_0_tensor = flow.tensor(feature_0_np, device="cuda", requires_grad=True)
    feature_1_tensor = flow.tensor(feature_1_np, device="cuda", requires_grad=True)
    if self_interaction:
        offset = 1
    else:
        offset = 0
    li = flow.tensor([i for i in range(27) for j in range(i + offset)])
    lj = flow.tensor([j for i in range(27) for j in range(i + offset)])
    T = flow.cat(
        [
            flow.reshape(feature_0_tensor, (batch_size, 1, embedding_size)),
            feature_1_tensor,
        ],
        dim=1,
    )
    Z = flow.matmul(T, T, transpose_b=True)
    # gather_nd not support half, so cast to float32
    Z = flow.cast(Z, flow.float32)
    Zflat = Z[:, li, lj]
    Zflat = flow.cast(Zflat, dtype)
    if output_concat:
        R = flow.cat([feature_0_tensor, Zflat], dim=1)
    else:
        R = Zflat
    if output_padding != 0:
        padding_tensor = flow.tensor(
            np.zeros((batch_size, output_padding)).astype(np_dtype),
            device="cuda",
            requires_grad=False,
        )
        R = flow.cat([R, padding_tensor], dim=1)
    loss = R.sum()
    loss.backward()

    fused_feature_0_tensor = flow.tensor(
        feature_0_np, device="cuda", requires_grad=True
    )
    fused_feature_1_tensor = flow.tensor(
        feature_1_np, device="cuda", requires_grad=True
    )
    if output_concat:
        output_concat_tensor = fused_feature_0_tensor
    else:
        output_concat_tensor = None
    fused_R = flow._C.fused_dot_feature_interaction(
        [
            fused_feature_0_tensor.reshape(batch_size, 1, embedding_size),
            fused_feature_1_tensor,
        ],
        output_concat=output_concat_tensor,
        self_interaction=self_interaction,
        output_padding=output_padding,
        pooling="none",
    )
    fused_loss = fused_R.sum()
    fused_loss.backward()
    test_case.assertTrue(
        np.allclose(
            feature_0_tensor.grad.numpy(),
            fused_feature_0_tensor.grad.numpy(),
            rtol=1e-3,
            atol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            feature_1_tensor.grad.numpy(),
            fused_feature_1_tensor.grad.numpy(),
            rtol=1e-3,
            atol=1e-4,
        )
    )

    test_case.assertTrue(np.allclose(fused_R.numpy(), R.numpy(), rtol=1e-3, atol=1e-3))


def _test_fused_dot_feature_interaction_pooling_sum(
    test_case, dtype, feature_dims, embedding_size, device_type="cuda",
):
    batch_size = 100
    if dtype == flow.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32

    feature_tensor_list = []
    fused_feature_tensor_list = []
    for dim in feature_dims:
        feature_np = np.random.uniform(-1, 1, (batch_size, dim, embedding_size)).astype(
            np_dtype
        )
        feature_tensor = flow.tensor(feature_np, device="cuda", requires_grad=True)
        feature_tensor_list.append(feature_tensor)
        fused_feature_tensor = flow.tensor(
            feature_np, device="cuda", requires_grad=True
        )
        fused_feature_tensor_list.append(fused_feature_tensor)

    concat = flow.cat(feature_tensor_list, dim=1,)
    if dtype == flow.float16:
        concat = flow.cast(concat, flow.float)
    sum_then_square = flow.sum(concat, dim=1) ** 2
    square_then_sum = flow.sum(concat ** 2, dim=1)
    bi_interaction = (sum_then_square - square_then_sum) * 0.5
    if dtype == flow.float16:
        bi_interaction = flow.cast(bi_interaction, flow.float16)
    R = flow.sum(bi_interaction, dim=-1, keepdim=True)
    loss = R.sum()
    loss.backward()

    fused_R = flow._C.fused_dot_feature_interaction(
        fused_feature_tensor_list, pooling="sum",
    )
    fused_loss = fused_R.sum()
    fused_loss.backward()
    if dtype == flow.float16:
        tol = 1e-2
    else:
        tol = 1e-3
    for i in range(len(feature_dims)):
        test_case.assertTrue(
            np.allclose(
                feature_tensor_list[i].grad.numpy(),
                fused_feature_tensor_list[i].grad.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )
        )
    test_case.assertTrue(np.allclose(fused_R.numpy(), R.numpy(), rtol=tol, atol=tol))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class FusedDotFeatureInteractionTestCase(flow.unittest.TestCase):
    def test_fused_dot_feature_interaction(test_case):
        arg_dict = OrderedDict()
        arg_dict["embedding_size"] = [128, 127, 16, 15]
        arg_dict["self_interaction"] = [False, True]
        arg_dict["output_concat"] = [True, False]
        arg_dict["output_padding"] = [1, 0]
        arg_dict["dtype"] = [flow.float16, flow.float32]
        for kwargs in GenArgDict(arg_dict):
            _test_fused_dot_feature_interaction(test_case, **kwargs)

    def test_fused_dot_feature_interaction_pooling_sum(test_case):
        arg_dict = OrderedDict()
        arg_dict["dtype"] = [flow.float16, flow.float32]
        arg_dict["feature_dims"] = [[39], [13, 26], [1, 10, 3]]
        arg_dict["embedding_size"] = [16, 11, 12]
        for kwargs in GenArgDict(arg_dict):
            _test_fused_dot_feature_interaction_pooling_sum(test_case, **kwargs)


if __name__ == "__main__":
    unittest.main()
