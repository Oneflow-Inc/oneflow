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

import numpy as np

import oneflow as flow
import oneflow.unittest


def _softmax(x):
    p = np.exp(x - x.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _test_sparse_softmax_cross_entropy(test_case, batch_size, num_classes):
    rng = np.random.RandomState(num_classes)
    np_x = rng.randn(batch_size, num_classes).astype(np.float16)
    np_labels = rng.randint(0, num_classes, size=(batch_size,))
    x = flow.tensor(np_x, device="cuda", requires_grad=True)
    labels = flow.tensor(np_labels, device="cuda", dtype=flow.int64)
    y = flow._C.sparse_softmax_cross_entropy(x, labels)
    y.sum().backward()

    p = _softmax(np_x.astype(np.float64))
    rows = np.arange(batch_size)
    ref_y = -np.log(p[rows, np_labels])
    ref_grad = p.copy()
    ref_grad[rows, np_labels] -= 1
    test_case.assertTrue(
        np.allclose(y.numpy().astype(np.float64), ref_y, rtol=1e-2, atol=1e-2)
    )
    test_case.assertTrue(
        np.allclose(x.grad.numpy().astype(np.float64), ref_grad, atol=2e-3)
    )


def _test_softmax_cross_entropy(test_case, batch_size, num_classes):
    rng = np.random.RandomState(num_classes)
    np_x = rng.randn(batch_size, num_classes).astype(np.float16)
    np_labels = np.eye(num_classes, dtype=np.float16)[
        rng.randint(0, num_classes, size=(batch_size,))
    ]
    x = flow.tensor(np_x, device="cuda", requires_grad=True)
    labels = flow.tensor(np_labels, device="cuda")
    y = flow._C.softmax_cross_entropy(x, labels)
    y.sum().backward()

    p = _softmax(np_x.astype(np.float64))
    ref_y = -(np_labels * np.log(p)).sum(axis=1)
    ref_grad = p - np_labels
    test_case.assertTrue(
        np.allclose(y.numpy().astype(np.float64), ref_y, rtol=1e-2, atol=1e-2)
    )
    test_case.assertTrue(
        np.allclose(x.grad.numpy().astype(np.float64), ref_grad, atol=2e-3)
    )


def _test_sparse_cross_entropy(test_case, batch_size, num_classes):
    rng = np.random.RandomState(num_classes)
    np_x = rng.uniform(0.1, 1.0, size=(batch_size, num_classes)).astype(np.float16)
    np_labels = rng.randint(0, num_classes, size=(batch_size,))
    x = flow.tensor(np_x, device="cuda", requires_grad=True)
    labels = flow.tensor(np_labels, device="cuda", dtype=flow.int64)
    y = flow._C.sparse_cross_entropy(x, labels, num_classes)
    y.sum().backward()

    rows = np.arange(batch_size)
    picked = np_x.astype(np.float64)[rows, np_labels]
    ref_y = -np.log(picked)
    ref_grad = np.zeros((batch_size, num_classes))
    ref_grad[rows, np_labels] = -1.0 / picked
    test_case.assertTrue(
        np.allclose(y.numpy().astype(np.float64), ref_y, rtol=1e-2, atol=1e-2)
    )
    test_case.assertTrue(
        np.allclose(x.grad.numpy().astype(np.float64), ref_grad, rtol=1e-2, atol=1e-2)
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyFloat16(flow.unittest.TestCase):
    # Odd and even class counts take the half and half2 gradient kernels.
    def test_sparse_softmax_cross_entropy(test_case):
        for num_classes in [10, 11, 1000]:
            _test_sparse_softmax_cross_entropy(test_case, 64, num_classes)

    def test_softmax_cross_entropy(test_case):
        for num_classes in [10, 11, 1000]:
            _test_softmax_cross_entropy(test_case, 64, num_classes)

    def test_sparse_cross_entropy(test_case):
        for num_classes in [10, 11, 1000]:
            _test_sparse_cross_entropy(test_case, 64, num_classes)


if __name__ == "__main__":
    unittest.main()
