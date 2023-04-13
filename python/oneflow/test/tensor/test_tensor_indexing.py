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
from oneflow.test_utils.test_util import GenArgList
from collections import OrderedDict
from oneflow.test_utils.automated_test_util import *

import numpy as np
import oneflow as flow
import oneflow.unittest


def _test_numpy_scalar_indexing(test_case, numpy_x, np_scalar):
    x = flow.Tensor(numpy_x)

    # basic_slice
    test_case.assertTrue(np.allclose(numpy_x[np_scalar(1)], x[np_scalar(1)].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[np_scalar(-2)], x[np_scalar(-2)].numpy()))
    test_case.assertTrue(
        np.allclose(
            numpy_x[np_scalar(0), np_scalar(1)], x[np_scalar(0), np_scalar(1)].numpy()
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[(np_scalar(0), np_scalar(1))],
            x[(np_scalar(0), np_scalar(1))].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[((np_scalar(0), np_scalar(1)))],
            x[((np_scalar(0), np_scalar(1)))].numpy(),
        )
    )


def _test_numpy_scalar_advance_indexing(test_case, numpy_x, np_scalar):
    x = flow.Tensor(numpy_x)

    # advance indexing
    test_case.assertTrue(
        np.allclose(
            numpy_x[[np_scalar(0), np_scalar(1)]],
            x[[np_scalar(0), np_scalar(1)]].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[[np_scalar(0), np_scalar(1)], [np_scalar(1), np_scalar(0)]],
            x[[np_scalar(0), np_scalar(1)], [np_scalar(1), np_scalar(0)]].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[
                [np_scalar(0), np_scalar(1)],
                [np_scalar(0), np_scalar(1)],
                [np_scalar(1), np_scalar(0)],
            ],
            x[
                [np_scalar(0), np_scalar(1)],
                [np_scalar(0), np_scalar(1)],
                [np_scalar(1), np_scalar(0)],
            ].numpy(),
        )
    )


def _test_basic_slice(test_case, numpy_x):
    x = flow.tensor(numpy_x)

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
            numpy_x[True, 1, ..., False, True], x[True, 1, ..., False, True].numpy()
        )
    )


# NOTE: When numpy>=1.23.0, the list of index will be seemed as basic indexing,
#       and tuple of index will be seemed as advanced indexing.
def _test_advanced_indexing(test_case, numpy_x):
    x = flow.tensor(numpy_x)

    test_case.assertTrue(np.allclose(numpy_x[[0, 1]], x[[0, 1]].numpy()))
    test_case.assertTrue(
        np.allclose(numpy_x[[0, 1], [1, 0]], x[[0, 1], [1, 0]].numpy())
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[tuple([[0, 1], [0, 1], [1, 0]])],
            x[[[0, 1], [0, 1], [1, 0]]].numpy(),
        )
    )
    test_case.assertTrue(np.allclose(numpy_x[tuple([[0], [1]])], x[[[0], [1]]].numpy()))
    test_case.assertTrue(
        np.allclose(
            numpy_x[tuple([[[0], [1]], [[0], [1]], [0, 1]])],
            x[[[[0], [1]], [[0], [1]], [0, 1]]].numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            numpy_x[tuple([[[0, 1], [1, 1]], [[0, 0], [1, 1]], [0, 1]])],
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
    mask = np.random.rand(numpy_x.shape[0], numpy_x.shape[1]).astype(np.float32)
    y = flow.tensor(mask)
    test_case.assertTrue(np.allclose(numpy_x[mask > 0.5], x[y > 0.5].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 0.5, 1], x[y > 0.5, 1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 0], x[y > 0].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 0, 1], x[y > 0, 1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 1], x[y > 1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 1, 1], x[y > 1, 1].numpy()))

    mask = np.random.rand(*numpy_x.shape).astype(np.float32)
    y = flow.tensor(mask)
    test_case.assertTrue(np.allclose(numpy_x[mask > 0.5], x[y > 0.5].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 0], x[y > 0].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 1], x[y > 1].numpy()))


def _test_advanced_indexing_array(test_case, numpy_x, dtype):
    x = flow.tensor(numpy_x)

    idx = np.array([0, 1], dtype=dtype)
    test_case.assertTrue(np.allclose(numpy_x[idx], x[idx].numpy()))

    idx1 = np.array([0, 1], dtype=dtype)
    idx2 = np.array([1, 0], dtype=dtype)
    test_case.assertTrue(np.allclose(numpy_x[idx1, idx2], x[idx1, idx2].numpy()))

    idx = np.array([[0, 1], [0, 1], [1, 0]], dtype=dtype)
    test_case.assertTrue(np.allclose(numpy_x[idx, :, :], x[idx, :, :].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[idx, idx, :], x[idx, idx, :].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[idx, idx, idx], x[idx, idx, idx].numpy()))

    idx1 = np.array([[1, 0, 1], [1, 1, 0]])
    idx2 = np.array([[0], [1]])
    test_case.assertTrue(
        np.allclose(numpy_x[:, idx1, :, idx2].shape, x[:, idx1, :, idx2].shape)
    )
    test_case.assertTrue(
        np.allclose(numpy_x[:, idx1, 1, idx2].shape, x[:, idx1, 1, idx2].shape)
    )
    test_case.assertTrue(
        np.allclose(numpy_x[idx1, :, idx2, :].shape, x[idx1, :, idx2, :].shape)
    )
    test_case.assertTrue(
        np.allclose(numpy_x[:, idx1, idx2, :].shape, x[:, idx1, idx2, :].shape)
    )


def _test_combining_indexing(test_case, numpy_x):
    x = flow.tensor(numpy_x)

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


def _test_mask_getitem(test_case, numpy_x):
    x = flow.tensor(numpy_x)

    mask = np.random.rand(*numpy_x.shape).astype(np.float32)
    y = flow.tensor(mask)
    test_case.assertTrue(np.allclose(numpy_x[mask > 0.5], x[y > 0.5].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 1.0], x[y > 1.0].numpy()))

    mask = np.random.rand(numpy_x.shape[0]).astype(np.float32)
    y = flow.tensor(mask)
    test_case.assertTrue(np.allclose(numpy_x[mask > 0.5], x[y > 0.5].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 1.0], x[y > 1.0].numpy()))

    test_case.assertTrue(np.allclose(numpy_x[mask > 0.5, 1], x[y > 0.5, 1].numpy()))
    test_case.assertTrue(np.allclose(numpy_x[mask > 1.0, 1], x[y > 1.0, 1].numpy()))


def _test_mask_setitem(test_case, numpy_x):
    x = flow.tensor(numpy_x)

    # mask tensor index
    mask = np.random.rand(*numpy_x.shape).astype(np.float32)
    y = flow.tensor(mask)

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


def _test_list_indexing_using_scalar_tensor(test_case, dtype):
    y = np.random.randint(0, 100, size=100)
    for i in range(len(y)):
        x = flow.tensor(i, dtype=dtype)
        test_case.assertEqual(y[i], y[x])


@flow.unittest.skip_unless_1n1d()
class TestTensorIndexing(flow.unittest.TestCase):
    def test_basic_slice(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        _test_basic_slice(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        _test_basic_slice(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        _test_basic_slice(test_case, numpy_x)

    def test_advanced_indexing(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        _test_advanced_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        _test_advanced_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        _test_advanced_indexing(test_case, numpy_x)

    def test_advanced_indexing_array(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 2, 2, 5]).astype(np.float32)
        _test_advanced_indexing_array(test_case, numpy_x, np.int32)
        _test_advanced_indexing_array(test_case, numpy_x, np.int64)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        _test_advanced_indexing_array(test_case, numpy_x, np.int32)
        _test_advanced_indexing_array(test_case, numpy_x, np.int64)

        numpy_x = np.arange(0, 720, 1).reshape([5, 8, 9, 2]).astype(np.float32)
        _test_advanced_indexing_array(test_case, numpy_x, np.int32)
        _test_advanced_indexing_array(test_case, numpy_x, np.int64)

    def test_combining_indexing(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        _test_combining_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        _test_combining_indexing(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        _test_combining_indexing(test_case, numpy_x)

    def test_numpy_scalar_indexing(test_case):
        for np_scalar in [np.int8, np.int16, np.int32, np.int64]:
            numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
            _test_numpy_scalar_indexing(test_case, numpy_x, np_scalar)

            numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
            _test_numpy_scalar_indexing(test_case, numpy_x, np_scalar)

            numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
            _test_numpy_scalar_indexing(test_case, numpy_x, np_scalar)

        # TODO: add np.int16 when advance indexing supports np.int16 mapping
        for np_scalar in [np.int32, np.int64]:
            numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
            _test_numpy_scalar_advance_indexing(test_case, numpy_x, np_scalar)

            numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
            _test_numpy_scalar_advance_indexing(test_case, numpy_x, np_scalar)

            numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
            _test_numpy_scalar_advance_indexing(test_case, numpy_x, np_scalar)

    def test_mask_getitem(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        _test_mask_getitem(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        _test_mask_getitem(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        _test_mask_getitem(test_case, numpy_x)

        numpy_x = np.arange(0, 27, 1).reshape(3, 3, 3)
        x = flow.tensor(numpy_x)
        test_case.assertTrue(
            np.allclose(
                numpy_x[[False, True, False], 1], x[[False, True, False], 1].numpy()
            )
        )
        test_case.assertTrue(
            np.allclose(
                numpy_x[[False, True, False], [True, False, False]],
                x[[False, True, False], [True, False, False]].numpy(),
            )
        )

    def test_mask_setitem(test_case):
        numpy_x = np.arange(0, 60, 1).reshape([3, 4, 5]).astype(np.float32)
        _test_mask_setitem(test_case, numpy_x)

        numpy_x = np.arange(0, 360, 1).reshape([3, 4, 5, 6]).astype(np.float32)
        _test_mask_setitem(test_case, numpy_x)

        numpy_x = np.arange(0, 720, 1).reshape([8, 9, 10]).astype(np.float32)
        _test_mask_setitem(test_case, numpy_x)

    def test_combined_mask_setitem(test_case):
        np_in = np.random.rand(5, 4, 3, 2)
        np_mask_dim1 = np.array([False, True, False, True])
        np_mask_dim3 = np.array([True, False])
        np_update = np.random.rand(2, 5, 3)
        np_in[:, np_mask_dim1, :, np_mask_dim3] = np_update

        flow_in = flow.tensor(np_in)
        flow_mask_dim1 = flow.tensor(np_mask_dim1)
        flow_mask_dim3 = flow.tensor(np_mask_dim3)
        flow_update = flow.tensor(np_update)
        flow_in[:, flow_mask_dim1, :, flow_mask_dim3] = flow_update
        test_case.assertTrue(np.array_equal(flow_in.numpy(), np_in))

    def test_non_contiguous_combined_mask_setitem(test_case):
        np_in = np.random.rand(5, 4, 3, 2)
        np_mask_dim1 = np.array([False, True, False])
        np_mask_dim3 = np.array([True, False, False, True, True])
        np_update = np.random.rand(4, 2, 3)

        flow_in = flow.tensor(np_in).permute(3, 2, 1, 0)  # (2, 3, 4, 5)
        flow_mask_dim1 = flow.tensor(np_mask_dim1)
        flow_mask_dim3 = flow.tensor(np_mask_dim3)
        flow_update = flow.tensor(np_update).permute(2, 1, 0)  # (3, 2, 4)
        flow_in[:, flow_mask_dim1, :, flow_mask_dim3] = flow_update

        np_in = np_in.transpose(3, 2, 1, 0)
        np_update = np_update.transpose(2, 1, 0)
        np_in[:, np_mask_dim1, :, np_mask_dim3] = np_update
        test_case.assertTrue(np.array_equal(flow_in.numpy(), np_in))

    def test_combined_indexing_setitem(test_case):
        np_in = np.random.rand(2, 3, 4)
        np_in[[0, 1], 1:2, [0, 1]] = 1.0

        flow_in = flow.tensor(np_in)
        flow_in[[0, 1], 1:2, [0, 1]] = 1.0
        test_case.assertTrue(np.array_equal(flow_in.numpy(), np_in))

    def test_expand_dim_setitem(test_case):
        a = flow.tensor(1.0)
        a[True, ...] = 0.0
        test_case.assertTrue(np.array_equal(a.numpy(), 0.0))

        a = flow.tensor(1.0)
        a[False, ...] = 1.0
        test_case.assertTrue(np.array_equal(a.numpy(), 1.0))

    def test_advanced_indexing_with_scalar_index(test_case):
        index = flow.tensor([0, 2])
        x = flow.randn(5)
        x[index[0]] = 1
        test_case.assertTrue(np.allclose(x[0].numpy(), 1))

    def test_list_indexing_using_scalar_tensor(test_case):
        arg_dict = OrderedDict()
        arg_dict["function_test"] = [
            _test_list_indexing_using_scalar_tensor,
        ]
        arg_dict["dtype"] = [flow.uint8, flow.int8, flow.int32, flow.int64]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=3, auto_backward=False)
    def test_advanced_indexing_with_0_size_tensor(test_case):
        device = random_device()
        data = torch.arange(8).reshape(2, 2, 2).to(device)
        ranges = []
        ranges.append(torch.ones(0, 1).to(torch.int64))
        ranges.append(torch.zeros(1, 3).to(torch.int64))
        res = data[ranges]
        return res

    @autotest(n=1)
    def test_dataloader_indexing_with_1_dim_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=512).to(device)
        batch_data = list()
        for i in range(512):
            batch_data.append(x[i])
        return torch.stack(batch_data)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_indecies_on_different_devices(test_case):
        x = flow.ones(3, 10)
        y = flow.ones(3, 10, device=flow.device("cuda:0"))

        x_idx = [flow.tensor([1, 2]), flow.tensor([2, 0], device=flow.device("cuda:0"))]
        y_idx = [flow.tensor([1, 2], device=flow.device("cuda:0")), flow.tensor([2, 0])]

        test_case.assertTrue(np.allclose(x[x_idx].numpy(), np.array([1, 1])))
        test_case.assertTrue(np.allclose(y[y_idx].numpy(), np.array([1, 1])))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensorIndexingMultiGpu(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_indecies_on_different_devices(test_case):
        x = flow.ones(3, 10, device=flow.device("cuda:0"))
        idx = [flow.tensor([1, 2], device=flow.device("cuda:1")), flow.tensor([2, 0])]
        test_case.assertTrue(np.allclose(x[idx].numpy(), np.array([1, 1])))


if __name__ == "__main__":
    unittest.main()
