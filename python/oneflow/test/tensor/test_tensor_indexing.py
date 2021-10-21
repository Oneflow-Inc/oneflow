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
import random
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest


def test_basic_slice(test_case, numpy_x):
    x = flow.Tensor(numpy_x)

    test_case.assertTrue(np.array_equal(numpy_x[1], x[1].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[-2], x[-2].numpy()))

    test_case.assertTrue(np.array_equal(numpy_x[0, 1], x[0, 1].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[(0, 1)], x[(0, 1)].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[((0, 1))], x[((0, 1))].numpy()))

    test_case.assertTrue(np.array_equal(numpy_x[None], x[None].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[True], x[True].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[1, None], x[1, None].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[1, None, 1], x[1, None, 1].numpy()))
    test_case.assertTrue(
        np.array_equal(numpy_x[1, None, None, 1], x[1, None, None, 1].numpy())
    )

    test_case.assertTrue(np.array_equal(numpy_x[:], x[:].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[:1], x[:1].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[0:1], x[0:1].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[-2:-1], x[-2:-1].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[2:100:200], x[2:100:200].numpy()))

    test_case.assertTrue(np.array_equal(numpy_x[0:2, ...], x[0:2, ...].numpy()))
    test_case.assertTrue(np.array_equal(numpy_x[0:2, ..., 1], x[0:2, ..., 1].numpy()))
    test_case.assertTrue(
        np.array_equal(numpy_x[0:2, ..., 1, 1], x[0:2, ..., 1, 1].numpy())
    )

    test_case.assertTrue(np.array_equal(numpy_x[0:4:2, ...], x[0:4:2, ...].numpy()))
    test_case.assertTrue(
        np.array_equal(numpy_x[0:2, None, ..., True], x[0:2, None, ..., True].numpy())
    )
    test_case.assertTrue(
        np.array_equal(
            numpy_x[None, ..., 0:4:2, True], x[None, ..., 0:4:2, True].numpy()
        )
    )


def test_advanced_indexing(test_case, numpy_x):
    x = flow.Tensor(numpy_x)

    test_case.assertTrue(np.array_equal(numpy_x[[0, 1]], x[[0, 1]].numpy()))
    test_case.assertTrue(
        np.array_equal(numpy_x[[0, 1], [1, 0]], x[[0, 1], [1, 0]].numpy())
    )
    test_case.assertTrue(
        np.array_equal(
            numpy_x[[[0, 1], [0, 1], [1, 0]]], x[[[0, 1], [0, 1], [1, 0]]].numpy()
        )
    )
    test_case.assertTrue(np.array_equal(numpy_x[[[0], [1]]], x[[[0], [1]]].numpy()))
    test_case.assertTrue(
        np.array_equal(
            numpy_x[[[[0], [1]], [[0], [1]], [0, 1]]],
            x[[[[0], [1]], [[0], [1]], [0, 1]]].numpy(),
        )
    )
    test_case.assertTrue(
        np.array_equal(
            numpy_x[[[[0, 1], [1, 1]], [[0, 0], [1, 1]], [0, 1]]],
            x[[[[0, 1], [1, 1]], [[0, 0], [1, 1]], [0, 1]]].numpy(),
        )
    )

    # Tensor index
    test_case.assertTrue(
        np.array_equal(
            numpy_x[np.array([0, 1]), np.array([1, 0])],
            x[flow.Tensor([0, 1]).long(), flow.Tensor([1, 0]).long()].numpy(),
        )
    )
    test_case.assertTrue(
        np.array_equal(
            numpy_x[:, np.array([[0, 1], [1, 1]]), np.array([[1, 0], [1, 1]])],
            x[
                :,
                flow.Tensor([[0, 1], [1, 1]]).long(),
                flow.Tensor([[1, 0], [1, 1]]).long(),
            ].numpy(),
        )
    )


def test_combining_indexing(test_case, numpy_x):
    x = flow.Tensor(numpy_x)

    test_case.assertTrue(
        np.array_equal(numpy_x[[0, 1], 1:2, [1, 0]], x[[0, 1], 1:2, [1, 0]].numpy())
    )
    test_case.assertTrue(
        np.array_equal(numpy_x[:, [0, 1], [1, 0]], x[:, [0, 1], [1, 0]].numpy())
    )
    test_case.assertTrue(np.array_equal(numpy_x[:, [0, 1], 1], x[:, [0, 1], 1].numpy()))
    test_case.assertTrue(
        np.array_equal(
            numpy_x[..., [0, 1], 1, [1, 0]], x[..., [0, 1], 1, [1, 0]].numpy()
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestTensorIndexing(flow.unittest.TestCase):
    def test_basic_slice(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5])
        test_basic_slice(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6])
        test_basic_slice(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10])
        test_basic_slice(test_case, numpy_x)

    def test_advanced_indexing(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5])
        test_advanced_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6])
        test_advanced_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10])
        test_advanced_indexing(test_case, numpy_x)

    def test_combining_indexing(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5])
        test_combining_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6])
        test_combining_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10])
        test_combining_indexing(test_case, numpy_x)


if __name__ == "__main__":
    unittest.main()
