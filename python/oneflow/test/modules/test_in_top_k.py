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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _topk_np(input, k, dim: int = -1, largest: bool = True, _sorted: bool = True):
    in_dims = input.shape
    out_dims = list(in_dims)
    num_axes = len(input.shape)
    if dim < 0:
        dim = dim + num_axes
    n = in_dims[dim]
    if k > n:
        k = n
    out_dims[dim] = k
    out_dims = tuple(out_dims)
    prev_dims = 1
    next_dims = 1
    for i in range(dim):
        prev_dims *= in_dims[i]
    for i in range(dim + 1, len(in_dims)):
        next_dims *= in_dims[i]
    input_flat = input.reshape((prev_dims, n, next_dims))
    values_ref = np.ndarray(shape=(prev_dims, k, next_dims), dtype=input.dtype)
    values_ref.fill(0)
    indices_ref = np.ndarray(shape=(prev_dims, k, next_dims), dtype=np.int64)
    indices_ref.fill(-1)
    for i in range(prev_dims):
        for j in range(next_dims):
            kv = []
            for x in range(n):
                val = input_flat[i, x, j]
                y = x * next_dims + i * in_dims[dim] * next_dims + j
                kv.append((val, x, y))
            cnt = 0
            for (val, x, y) in sorted(kv, key=lambda x: (x[0], -x[1]), reverse=largest):
                values_ref[i, cnt, j] = val
                indices_ref[i, cnt, j] = x
                cnt += 1
                if cnt >= k or cnt >= n:
                    break
    values_ref = values_ref.reshape(out_dims)
    indices_ref = indices_ref.reshape(out_dims)
    return (values_ref, indices_ref)


def _in_top_k_np(targets, predictions, k):
    assert (
        targets.shape[0] == predictions.shape[0]
    ), "The num of targets must equal the num of predictions"
    assert len(targets.shape) == 1, "The dimension of targets must be 1"
    assert len(predictions.shape) == 2, "The dimension of predictions must be 2"
    results = np.zeros_like(targets, dtype=np.int8)
    for i in range(len(results)):
        (_, indices_topk) = _topk_np(predictions[i], k)
        if targets[i] in indices_topk:
            results[i] = 1
    return results


def _test_in_top_k_impl(test_case, shape, k, device):
    np_targets = np.random.randint(0, shape[1], size=shape[0])
    np_predictions = np.random.rand(*shape)
    of_targets = flow.tensor(
        np_targets, dtype=flow.int32, device=flow.device(device), requires_grad=False
    )
    of_predictions = flow.tensor(
        np_predictions,
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.in_top_k(of_targets, of_predictions, k)
    np_out = _in_top_k_np(np_targets, np_predictions, k)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestInTopK(flow.unittest.TestCase):
    def test_in_top_k(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (3, 4), (5, 6)]
        arg_dict["k"] = [1, 2, 5]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_in_top_k_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
