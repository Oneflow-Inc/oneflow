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
import random
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest


def test_basic_slice(test_case, numpy_x, x):
    test_case.assertTrue(np.allclose(numpy_x[1], x[1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[-2], x[-2].numpy()))

    test_case.assertTrue(np.allclose(numpy_x[0, 1], x[0, 1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[(0, 1)], x[(0, 1)].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[((0, 1))], x[((0, 1))].numpy()))

    test_case.assertTrue(np.allclose(numpy_x[None], x[None].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[True], x[True].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[1, None], x[1, None].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[1, None, 1], x[1, None, 1].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[1, None, None, 1], x[1, None, None, 1].numpy())
    )

    test_case.assertTrue(np.allclose(numpy_x[:], x[:].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[:1], x[:1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[0:1], x[0:1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[-2:-1], x[-2:-1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[2:100:200], x[2:100:200].numpy()))

    test_case.assertTrue(np.allclose(numpy_x[0:2, ...], x[0:2, ...].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[0:2, ..., 1], x[0:2, ..., 1].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[0:2, ..., 1, 1], x[0:2, ..., 1, 1].numpy())
    )

    test_case.assertTrue(np.allclose(numpy_x[0:4:2, ...], x[0:4:2, ...].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[0:2, None, ..., True], x[0:2, None, ..., True].numpy())
    )
    test_case.assertTrue(
        np.allclose(numpy_x[None, ..., 0:4:2, True], x[None, ..., 0:4:2, True].numpy())
    )

    test_case.assertTrue(np.allclose(numpy_x[False, ...], x[False, ...].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[False, True, ...], x[False, True, ...].numpy())
    )
    test_case.assertTrue(
        np.allclose(numpy_x[True, ..., False, True], x[True, ..., False, True].numpy())
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[True, None, ..., False, True],
            x[True, None, ..., False, True].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[True, 1, ..., False, True], x[True, 1, ..., True, True].numpy()
        )
    )


def test_advanced_indexing(test_case, numpy_x, x):
    test_case.assertTrue(np.allclose(numpy_x[[0, 1]], x[[0, 1]].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[[0, 1], [1, 0]], x[[0, 1], [1, 0]].numpy())
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[[[0, 1], [0, 1], [1, 0]]], x[[[0, 1], [0, 1], [1, 0]]].numpy()
        )
    )
    test_case.assertTrue(np.allclose(numpy_x[[[0], [1]]], x[[[0], [1]]].numpy()))
    test_case.assertTrue(
        np.allclose(
            numpy_x[[[[0], [1]], [[0], [1]], [0, 1]]],
            x[[[[0], [1]], [[0], [1]], [0, 1]]].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[[[[0, 1], [1, 1]], [[0, 0], [1, 1]], [0, 1]]],
            x[[[[0, 1], [1, 1]], [[0, 0], [1, 1]], [0, 1]]].numpy(),
        )
    )

    # Tensor index
    test_case.assertTrue(
        np.allclose(
            numpy_x[np.array([0, 1]), np.array([1, 0])],
            x[flow.tensor([0, 1]), flow.tensor([1, 0])].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[:, np.array([[0, 1], [1, 1]]), np.array([[1, 0], [1, 1]])],
            x[:, flow.tensor([[0, 1], [1, 1]]), flow.tensor([[1, 0], [1, 1]]),].numpy(),
        )
    )

    # mask tensor index
    # mask = np.random.rand(numpy_x.shape[0], numpy_x.shape[1]).astype(np.float32)
    # y = flow.tensor(mask)
    # print(numpy_x[mask > 0.5])
    # print(x[y > 0.5].to_local().numpy())
    # test_case.assertTrue(np.allclose(numpy_x[mask > 0.5], x[y > 0.5].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 0.5, 1], x[y > 0.5, 1].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 0], x[y > 0].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 0, 1], x[y > 0, 1].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 1], x[y > 1].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 1, 1], x[y > 1, 1].numpy()))

    # mask = np.random.rand(*numpy_x.shape).astype(np.float32)
    # y = flow.tensor(mask)
    # test_case.assertTrue(np.allclose(numpy_x[mask > 0.5], x[y > 0.5].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 0], x[y > 0].numpy()))
    # test_case.assertTrue(np.allclose(numpy_x[mask > 1], x[y > 1].numpy()))


def test_advanced_indexing_array(test_case, numpy_x, x, dtype):
    idx = np.array([0, 1], dtype=dtype)
    test_case.assertTrue(np.allclose(numpy_x[idx], x[idx].numpy()))

    idx1 = np.array([0, 1], dtype=dtype)
    idx2 = np.array([1, 0], dtype=dtype)
    test_case.assertTrue(np.allclose(numpy_x[idx1, idx2], x[idx1, idx2].numpy()))

    idx = np.array([[0, 1], [0, 1], [1, 0]], dtype=dtype)
    test_case.assertTrue(np.allclose(numpy_x[idx, :, :], x[idx, :, :].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[idx, idx, :], x[idx, idx, :].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[idx, idx, idx], x[idx, idx, idx].numpy()))


def test_combining_indexing(test_case, numpy_x, x):
    test_case.assertTrue(
        np.allclose(numpy_x[[0, 1], 1:2, [1, 0]], x[[0, 1], 1:2, [1, 0]].numpy())
    )
    test_case.assertTrue(
        np.allclose(numpy_x[:, [0, 1], [1, 0]], x[:, [0, 1], [1, 0]].numpy())
    )
    test_case.assertTrue(np.allclose(numpy_x[:, [0, 1], 1], x[:, [0, 1], 1].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[..., [0, 1], 1, [1, 0]], x[..., [0, 1], 1, [1, 0]].numpy())
    )


def test_mask_setitem(test_case, numpy_x, x):
    # mask tensor index
    mask = np.random.rand(*numpy_x.shape).astype(np.float32)
    y = flow.tensor(mask, device="cuda").to_consistent(placement=x.placement, sbp=x.sbp)

    # broadcast set
    x[y > 0.5] = 1.0
    numpy_x[mask > 0.5] = 1.0
    test_case.assertTrue(np.allclose(numpy_x, x.numpy()))

    # elementwise set
    update = np.random.randn((mask > 0.5).sum()).astype(np.float32)
    tensor_update = flow.tensor(update)
    x[y > 0.5] = tensor_update
    numpy_x[mask > 0.5] = update
    test_case.assertTrue(np.allclose(numpy_x, x.numpy()))

    # empty mask
    x[y > 1.0] = 1.0
    numpy_x[mask > 1.0] = 1.0
    test_case.assertTrue(np.allclose(numpy_x, x.numpy()))


@flow.unittest.skip_unless_1n2d()
class TestConsistentTensorIndexing(flow.unittest.TestCase):
    def test_basic_slice(test_case):
        placement = flow.placement("cuda",{0:[0, 1]})
        sbp = sbp = flow.sbp.broadcast

        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_basic_slice(test_case, numpy_x, x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_basic_slice(test_case, numpy_x, x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_basic_slice(test_case, numpy_x, x)

    def test_advanced_indexing(test_case):
        placement = flow.placement("cuda",{0:[0, 1]})
        sbp = sbp = flow.sbp.broadcast
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing(test_case, numpy_x, x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing(test_case, numpy_x, x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing(test_case, numpy_x, x)

    def test_advanced_indexing_array(test_case):
        placement = flow.placement("cuda",{0:[0, 1]})
        sbp = sbp = flow.sbp.broadcast

        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing_array(test_case, numpy_x, x, np.int32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing_array(test_case, numpy_x, x, np.int64)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing_array(test_case, numpy_x, x, np.int32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_advanced_indexing_array(test_case, numpy_x, x, np.int64)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        x = flow.tensor(numpy_x, placement=placement, sbp=sbp)
        test_advanced_indexing_array(test_case, numpy_x, x, np.int32)
        x = flow.tensor(numpy_x, placement=placement, sbp=sbp)
        test_advanced_indexing_array(test_case, numpy_x, x, np.int64)

    def test_combining_indexing(test_case):
        placement = flow.placement("cuda",{0:[0, 1]})
        sbp = sbp = flow.sbp.broadcast

        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_combining_indexing(test_case, numpy_x, x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_combining_indexing(test_case, numpy_x, x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        x = flow.tensor(numpy_x)
        x = x.to_consistent(placement=placement, sbp=sbp)
        test_combining_indexing(test_case, numpy_x, x)

    # def test_mask_setitem(test_case):
    #     placement = flow.placement("cuda",{0:[0, 1]})
    #     sbp = sbp = flow.sbp.broadcast

    #     numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
    #     x = flow.tensor(numpy_x)
    #     x = x.to_consistent(placement=placement, sbp=sbp)
    #     test_mask_setitem(test_case, numpy_x, x)

    #     numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
    #     x = flow.tensor(numpy_x)
    #     x = x.to_consistent(placement=placement, sbp=sbp)
    #     test_mask_setitem(test_case, numpy_x, x)

    #     numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
    #     x = flow.tensor(numpy_x)
    #     x = x.to_consistent(placement=placement, sbp=sbp)
    #     test_mask_setitem(test_case, numpy_x, x)


if __name__ == "__main__":
    unittest.main()
