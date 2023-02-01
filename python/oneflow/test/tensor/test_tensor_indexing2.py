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

# This test code is referenced from: https://github.com/pytorch/pytorch/blob/cd41c8f032dd06c445bf97fc76fb82008b19afcb/test/test_indexing.py

from collections import OrderedDict
import random
from random import randrange
import unittest

import numpy as np

import oneflow as flow
from oneflow.test_utils.test_util import GenArgDict
import oneflow.unittest


def _assert_tensor_equal(test_case, tensor1, tensor2, atol=0.0, rtol=0.0):
    test_case.assertTrue(np.allclose(tensor1.numpy(), tensor2.numpy()))


def consec(size, start=1):
    """
    Generate a arithmetic progression with given size and start value.
    """
    sequence = flow.ones([int(np.array(size).prod(0)),]).cumsum(0)
    sequence.add_(start - 1)
    return sequence.view(*size)


def _test_basic_slice(test_case, device, dtype):
    reference = consec((3, 3, 3)).to(device=device, dtype=dtype)

    # empty tensor indexing
    _assert_tensor_equal(
        test_case,
        reference[flow.LongTensor().to(device)],
        flow.empty(0, 3, 3),
        atol=0,
        rtol=0,
    )

    _assert_tensor_equal(test_case, reference[0], consec((3, 3)), atol=0, rtol=0)
    _assert_tensor_equal(test_case, reference[1], consec((3, 3), 10), atol=0, rtol=0)
    _assert_tensor_equal(test_case, reference[2], consec((3, 3), 19), atol=0, rtol=0)
    _assert_tensor_equal(test_case, reference[0, 1], consec((3,), 4), atol=0, rtol=0)
    _assert_tensor_equal(test_case, reference[0:2], consec((2, 3, 3)), atol=0, rtol=0)
    test_case.assertEqual(reference[2, 2, 2].item(), 27)
    _assert_tensor_equal(test_case, reference[:], consec((3, 3, 3)), atol=0, rtol=0)

    # indexing with Ellipsis
    _assert_tensor_equal(
        test_case,
        reference[..., 2],
        flow.tensor([[3.0, 6.0, 9.0], [12.0, 15.0, 18.0], [21.0, 24.0, 27.0]]),
        atol=0,
        rtol=0,
    )
    _assert_tensor_equal(
        test_case, reference[0, ..., 2], flow.tensor([3.0, 6.0, 9.0]), atol=0, rtol=0
    )
    _assert_tensor_equal(
        test_case, reference[..., 2], reference[:, :, 2], atol=0, rtol=0
    )
    _assert_tensor_equal(
        test_case, reference[0, ..., 2], reference[0, :, 2], atol=0, rtol=0
    )
    _assert_tensor_equal(
        test_case, reference[0, 2, ...], reference[0, 2], atol=0, rtol=0
    )
    test_case.assertEqual(reference[..., 2, 2, 2].item(), 27)
    test_case.assertEqual(reference[2, ..., 2, 2].item(), 27)
    test_case.assertEqual(reference[2, 2, ..., 2].item(), 27)
    test_case.assertEqual(reference[2, 2, 2, ...].item(), 27)
    _assert_tensor_equal(test_case, reference[...], reference, atol=0, rtol=0)

    reference_5d = consec((3, 3, 3, 3, 3)).to(device)
    _assert_tensor_equal(
        test_case, reference_5d[..., 1, 0], reference_5d[:, :, :, 1, 0], atol=0, rtol=0
    )
    _assert_tensor_equal(
        test_case,
        reference_5d[2, ..., 1, 0],
        reference_5d[2, :, :, 1, 0],
        atol=0,
        rtol=0,
    )
    _assert_tensor_equal(
        test_case,
        reference_5d[2, 1, 0, ..., 1],
        reference_5d[2, 1, 0, :, 1],
        atol=0,
        rtol=0,
    )
    _assert_tensor_equal(test_case, reference_5d[...], reference_5d, atol=0, rtol=0)

    # LongTensor indexing
    reference = consec((5, 5, 5)).to(device=device, dtype=dtype)
    idx = flow.LongTensor([2, 4]).to(device)
    _assert_tensor_equal(
        test_case, reference[idx], flow.stack([reference[2], reference[4]])
    )

    # None indexing
    _assert_tensor_equal(test_case, reference[2, None], reference[2].unsqueeze(0))
    _assert_tensor_equal(
        test_case, reference[2, None, None], reference[2].unsqueeze(0).unsqueeze(0)
    )
    _assert_tensor_equal(test_case, reference[2:4, None], reference[2:4].unsqueeze(1))
    _assert_tensor_equal(
        test_case,
        reference[None, 2, None, None],
        reference.unsqueeze(0)[:, 2].unsqueeze(0).unsqueeze(0),
    )
    _assert_tensor_equal(
        test_case,
        reference[None, 2:5, None, None],
        reference.unsqueeze(0)[:, 2:5].unsqueeze(2).unsqueeze(2),
    )

    # indexing 0-length slice
    _assert_tensor_equal(test_case, flow.empty(0, 5, 5), reference[slice(0)])
    _assert_tensor_equal(test_case, flow.empty(0, 5), reference[slice(0), 2])
    _assert_tensor_equal(test_case, flow.empty(0, 5), reference[2, slice(0)])
    _assert_tensor_equal(test_case, flow.tensor([]), reference[2, 1:1, 2])

    # indexing with step
    reference = consec((10, 10, 10)).to(device=device, dtype=dtype)
    _assert_tensor_equal(
        test_case, reference[1:5:2], flow.stack([reference[1], reference[3]], 0)
    )
    _assert_tensor_equal(
        test_case,
        reference[1:6:2],
        flow.stack([reference[1], reference[3], reference[5]], 0),
    )
    _assert_tensor_equal(
        test_case, reference[1:9:4], flow.stack([reference[1], reference[5]], 0)
    )
    _assert_tensor_equal(
        test_case,
        reference[2:4, 1:5:2],
        flow.stack([reference[2:4, 1], reference[2:4, 3]], 1),
    )
    _assert_tensor_equal(
        test_case,
        reference[3, 1:6:2],
        flow.stack([reference[3, 1], reference[3, 3], reference[3, 5]], 0),
    )
    _assert_tensor_equal(
        test_case,
        reference[None, 2, 1:9:4],
        flow.stack([reference[2, 1], reference[2, 5]], 0).unsqueeze(0),
    )
    _assert_tensor_equal(
        test_case,
        reference[:, 2, 1:6:2],
        flow.stack([reference[:, 2, 1], reference[:, 2, 3], reference[:, 2, 5]], 1),
    )

    lst = [list(range(i, i + 10)) for i in range(0, 100, 10)]
    tensor = flow.DoubleTensor(lst).to(device=device, dtype=dtype)
    for _ in range(10):
        idx1_start = randrange(10)
        idx1_end = idx1_start + randrange(1, 10 - idx1_start + 1)
        idx1_step = randrange(1, 8)
        idx1 = slice(idx1_start, idx1_end, idx1_step)
        if randrange(2) == 0:
            idx2_start = randrange(10)
            idx2_end = idx2_start + randrange(1, 10 - idx2_start + 1)
            idx2_step = randrange(1, 8)
            idx2 = slice(idx2_start, idx2_end, idx2_step)
            lst_indexed = [l[idx2] for l in lst[idx1]]
            tensor_indexed = tensor[idx1, idx2]
        else:
            lst_indexed = lst[idx1]
            tensor_indexed = tensor[idx1]
        _assert_tensor_equal(
            test_case, flow.DoubleTensor(lst_indexed).to(dtype), tensor_indexed
        )

    test_case.assertRaises(RuntimeError, lambda: reference[1:9:0])
    test_case.assertRaises(RuntimeError, lambda: reference[1:9:-1])

    test_case.assertRaises(IndexError, lambda: reference[1, 1, 1, 1])
    test_case.assertRaises(IndexError, lambda: reference[1, 1, 1, 1:1])
    test_case.assertRaises(IndexError, lambda: reference[3, 3, 3, 3, 3, 3, 3, 3])

    test_case.assertRaises(IndexError, lambda: reference[0.0])
    test_case.assertRaises(RuntimeError, lambda: reference[0.0:2.0])
    test_case.assertRaises(IndexError, lambda: reference[0.0, 0.0:2.0])
    test_case.assertRaises(IndexError, lambda: reference[0.0, :, 0.0:2.0])
    test_case.assertRaises(IndexError, lambda: reference[0.0, ..., 0.0:2.0])
    test_case.assertRaises(IndexError, lambda: reference[0.0, :, 0.0])


def _test_advanced_indexing(test_case, device, dtype):
    # pick a random valid indexer type
    def ri(indices):
        choice = random.randint(0, 2)
        if choice == 0:
            return flow.LongTensor(indices).to(device)
        elif choice == 1:
            return list(indices)
        else:
            return tuple(indices)

    def validate_indexing(x):
        _assert_tensor_equal(test_case, x[[0]], consec((1,)))
        _assert_tensor_equal(test_case, x[ri([0]),], consec((1,)))
        _assert_tensor_equal(test_case, x[ri([3]),], consec((1,), 4))
        _assert_tensor_equal(test_case, x[[2, 3, 4]], consec((3,), 3))
        _assert_tensor_equal(test_case, x[ri([2, 3, 4]),], consec((3,), 3))
        _assert_tensor_equal(
            test_case,
            x[ri([0, 2, 4]),],
            flow.tensor([1, 3, 5], dtype=dtype, device=device),
        )

    def validate_setting(x):
        x[[0]] = -2
        _assert_tensor_equal(
            test_case, x[[0]], flow.tensor([-2], dtype=dtype, device=device)
        )
        x[[0]] = -1
        _assert_tensor_equal(
            test_case, x[ri([0]),], flow.tensor([-1], dtype=dtype, device=device)
        )
        x[[2, 3, 4]] = 4
        _assert_tensor_equal(
            test_case, x[[2, 3, 4]], flow.tensor([4, 4, 4], dtype=dtype, device=device)
        )
        x[ri([2, 3, 4]),] = 3
        _assert_tensor_equal(
            test_case,
            x[ri([2, 3, 4]),],
            flow.tensor([3, 3, 3], dtype=dtype, device=device),
        )
        x[ri([0, 2, 4]),] = flow.tensor([5, 4, 3], dtype=dtype, device=device)
        _assert_tensor_equal(
            test_case,
            x[ri([0, 2, 4]),],
            flow.tensor([5, 4, 3], dtype=dtype, device=device),
        )

    # 1d tensor and integer index setitem and getitem
    reference = consec((10,)).to(device=device, dtype=dtype)
    validate_indexing(reference)
    validate_setting(reference)

    # reference is 1 2
    #              3 4
    #              5 6
    reference = consec((3, 2)).to(device=device, dtype=dtype)
    _assert_tensor_equal(
        test_case,
        reference[ri([0, 1, 2]), ri([0])],
        flow.tensor([1, 3, 5], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([0, 1, 2]), ri([1])],
        flow.tensor([2, 4, 6], dtype=dtype, device=device),
    )
    _assert_tensor_equal(test_case, reference[ri([0]), ri([0])], consec((1,)))
    _assert_tensor_equal(test_case, reference[ri([2]), ri([1])], consec((1,), 6))
    _assert_tensor_equal(
        test_case,
        reference[[ri([0, 0]), ri([0, 1])]],
        flow.tensor([1, 2], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[[ri([0, 1, 1, 0, 2]), ri([1])]],
        flow.tensor([2, 4, 4, 2, 6], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]],
        flow.tensor([1, 2, 3, 3], dtype=dtype, device=device),
    )

    rows = ri([[0, 0], [1, 2]])
    columns = ([0],)
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[1, 1], [3, 5]], dtype=dtype, device=device),
    )

    rows = ri([[0, 0], [1, 2]])
    columns = ri([1, 0])
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[2, 1], [4, 5]], dtype=dtype, device=device),
    )
    rows = ri([[0, 0], [1, 2]])
    columns = ri([[0, 1], [1, 0]])
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[1, 2], [4, 5]], dtype=dtype, device=device),
    )

    # setting values
    reference[ri([0]), ri([1])] = -1
    _assert_tensor_equal(
        test_case,
        reference[ri([0]), ri([1])],
        flow.tensor([-1], dtype=dtype, device=device),
    )
    reference[ri([0, 1, 2]), ri([0])] = flow.tensor(
        [-1, 2, -4], dtype=dtype, device=device
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([0, 1, 2]), ri([0])],
        flow.tensor([-1, 2, -4], dtype=dtype, device=device),
    )
    reference[rows, columns] = flow.tensor([[4, 6], [2, 3]], dtype=dtype, device=device)
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[4, 6], [2, 3]], dtype=dtype, device=device),
    )

    # Test non-contiguous(by transpose) reference
    # Transposed: [[0, 4, 8],
    #              [1, 5, 9],
    #              [2, 6, 10],
    #              [3, 7, 11]]
    reference = flow.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=dtype, device=device
    ).T

    _assert_tensor_equal(
        test_case,
        reference[ri([0, 1, 2]), ri([0])],
        flow.tensor([0, 1, 2], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([0, 1, 2]), ri([1])],
        flow.tensor([4, 5, 6], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([0]), ri([0])],
        flow.tensor([0], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([2]), ri([1])],
        flow.tensor([6], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[[ri([0, 0]), ri([0, 1])]],
        flow.tensor([0, 4], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[[ri([0, 1, 1, 0, 3]), ri([1])]],
        flow.tensor([4, 5, 5, 4, 7], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]],
        flow.tensor([0, 4, 1, 1], dtype=dtype, device=device),
    )

    rows = ri([[0, 0], [1, 2]])
    columns = ([0],)
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[0, 0], [1, 2]], dtype=dtype, device=device),
    )

    rows = ri([[0, 0], [1, 2]])
    columns = ri([1, 0])
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[4, 0], [5, 2]], dtype=dtype, device=device),
    )
    rows = ri([[0, 0], [1, 3]])
    columns = ri([[0, 1], [1, 2]])
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[0, 4], [5, 11]], dtype=dtype, device=device),
    )

    # setting values
    reference[ri([0]), ri([1])] = -1
    _assert_tensor_equal(
        test_case,
        reference[ri([0]), ri([1])],
        flow.tensor([-1], dtype=dtype, device=device),
    )
    reference[ri([0, 1, 2]), ri([0])] = flow.tensor(
        [-1, 2, -4], dtype=dtype, device=device
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([0, 1, 2]), ri([0])],
        flow.tensor([-1, 2, -4], dtype=dtype, device=device),
    )
    reference[rows, columns] = flow.tensor([[4, 6], [2, 3]], dtype=dtype, device=device)
    _assert_tensor_equal(
        test_case,
        reference[rows, columns],
        flow.tensor([[4, 6], [2, 3]], dtype=dtype, device=device),
    )

    # Tests using less than the number of dims, and ellipsis
    # reference is 1 2
    #              3 4
    #              5 6
    reference = consec((3, 2)).to(dtype=dtype, device=device)
    _assert_tensor_equal(
        test_case,
        reference[ri([0, 2]),],
        flow.tensor([[1, 2], [5, 6]], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[ri([1]), ...],
        flow.tensor([[3, 4]], dtype=dtype, device=device),
    )
    _assert_tensor_equal(
        test_case,
        reference[..., ri([1])],
        flow.tensor([[2], [4], [6]], dtype=dtype, device=device),
    )

    # verify too many indices fails
    with test_case.assertRaises(IndexError):
        reference[ri([1]), ri([0, 2]), ri([3])]

    # test invalid index fails
    reference = flow.empty(10, dtype=dtype, device=device)
    for err_idx in (10, -11):
        with test_case.assertRaisesRegex(IndexError, r"out of range"):
            reference[err_idx]


def _test_combined_indexing(test_case, device, dtype):
    def tensor_indices_to_np(tensor, indices):
        # convert the flow Tensor to a numpy array
        tensor = tensor.to(device="cpu")
        npt = tensor.numpy()

        # convert indices
        idxs = tuple(
            i.tolist() if isinstance(i, flow.LongTensor) else i for i in indices
        )

        return npt, idxs

    def get_numpy(tensor, indices):
        npt, idxs = tensor_indices_to_np(tensor, indices)

        # index and return as a flow Tensor
        return flow.tensor(npt[idxs], dtype=dtype, device=device)

    def set_numpy(tensor, indices, value):
        if not isinstance(value, int):
            if device != "cpu":
                value = value.cpu()
            value = value.numpy()

        npt, idxs = tensor_indices_to_np(tensor, indices)
        npt[idxs] = value
        return npt

    def assert_get_eq(tensor, indexer):
        _assert_tensor_equal(test_case, tensor[indexer], get_numpy(tensor, indexer))

    def assert_set_eq(tensor, indexer, val):
        pyt = tensor.clone()
        np_ref = tensor.clone()
        pyt[indexer] = val
        np_ref = flow.tensor(
            set_numpy(np_ref, indexer, val), dtype=dtype, device=device
        )
        _assert_tensor_equal(test_case, pyt, np_ref)

    def assert_backward_eq(tensor, indexer):
        cpu = tensor.cpu().float().clone().detach().requires_grad_(True)
        outcpu = cpu[indexer]
        grad = flow.rand(outcpu.shape)
        outcpu.backward(grad)
        dev = cpu.to(device).detach().requires_grad_(True)
        outdev = dev[indexer]
        outdev.backward(grad.to(device))
        _assert_tensor_equal(test_case, cpu.grad, dev.grad)

    def get_set_tensor(indexed, indexer):
        set_size = indexed[indexer].size()
        set_count = indexed[indexer].numel()
        set_tensor = flow.randperm(set_count).view(set_size).to(dtype).to(device)
        return set_tensor

    # Tensor is  0  1  2  3  4
    #            5  6  7  8  9
    #           10 11 12 13 14
    #           15 16 17 18 19
    reference = flow.arange(0.0, 20, device=device).to(dtype).view(4, 5)

    indices_to_test = [
        # grab the second, fourth columns
        [slice(None), [1, 3]],
        # first, third rows,
        [[0, 2], slice(None)],
        # TODO(wyg): only support getitem but not setitem
        #  # weird shape
        #  [slice(None), [[0, 1],
        #                 [2, 3]]],
        # negatives
        [[-1], [0]],
        [[0, 2], [-1]],
        [slice(None), [-1]],
    ]

    # test getitem
    get_indices_to_test = indices_to_test + [[slice(None), [0, 1, 1, 2, 2]]]
    get_indices_to_test = indices_to_test + [
        [slice(None), [[0, 1], [2, 3]]]
    ]  # TODO: test setitem
    for indexer in get_indices_to_test:
        assert_get_eq(reference, indexer)
        if device != "cpu":
            assert_backward_eq(reference, indexer)

    # test setitem
    for indexer in indices_to_test:
        assert_set_eq(reference, indexer, 44)
        assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))

    #########################
    # test more dims tensor #
    #########################
    reference = flow.arange(0.0, 160, device=device).to(dtype).view(4, 8, 5)

    indices_to_test = [
        [slice(None), slice(None), [0, 3, 4]],
        [slice(None), [2, 4, 5, 7], slice(None)],
        [[2, 3], slice(None), slice(None)],
        [slice(None), [0, 2, 3], [1, 3, 4]],
        [slice(None), [0], [1, 2, 4]],
        [slice(None), [0, 1, 3], [4]],
        [slice(None), [[0, 1], [1, 0]], [[2, 3]]],
        [slice(None), [[0, 1], [2, 3]], [[0]]],
        [slice(None), [[5, 6]], [[0, 3], [4, 4]]],
        [[0, 2, 3], [1, 3, 4], slice(None)],
        [[0], [1, 2, 4], slice(None)],
        [[0, 1, 3], [4], slice(None)],
        [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
        [[[0, 1], [1, 0]], [[2, 3]], slice(None)],
        [[[0, 1], [2, 3]], [[0]], slice(None)],
        [[[2, 1]], [[0, 3], [4, 4]], slice(None)],
        [[[2]], [[0, 3], [4, 1]], slice(None)],
        # non-contiguous indexing subspace
        [[0, 2, 3], slice(None), [1, 3, 4]],
        # less dim, ellipsis
        [[0, 2],],
        [[0, 2], slice(None)],
        [[0, 2], Ellipsis],
        [[0, 2], slice(None), Ellipsis],
        [[0, 2], Ellipsis, slice(None)],
        [[0, 2], [1, 3]],
        [[0, 2], [1, 3], Ellipsis],
        [Ellipsis, [1, 3], [2, 3]],
        [Ellipsis, [2, 3, 4]],
        [Ellipsis, slice(None), [2, 3, 4]],
        [slice(None), Ellipsis, [2, 3, 4]],
        # ellipsis counts for nothing
        [Ellipsis, slice(None), slice(None), [0, 3, 4]],
        [slice(None), Ellipsis, slice(None), [0, 3, 4]],
        [slice(None), slice(None), Ellipsis, [0, 3, 4]],
        [slice(None), slice(None), [0, 3, 4], Ellipsis],
        [Ellipsis, [[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
        [[[0, 1], [1, 0]], [[2, 1], [3, 5]], Ellipsis, slice(None)],
        [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None), Ellipsis],
    ]

    for indexer in indices_to_test:
        assert_get_eq(reference, indexer)
        assert_set_eq(reference, indexer, 212)
        assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
        if device != "cpu":
            assert_backward_eq(reference, indexer)

    reference = flow.arange(0.0, 1296, device=device).to(dtype).view(3, 9, 8, 6)

    indices_to_test = [
        [slice(None), slice(None), slice(None), [0, 3, 4]],
        [slice(None), slice(None), [2, 4, 5, 7], slice(None)],
        [slice(None), [2, 3], slice(None), slice(None)],
        [[1, 2], slice(None), slice(None), slice(None)],
        [slice(None), slice(None), [0, 2, 3], [1, 3, 4]],
        [slice(None), slice(None), [0], [1, 2, 4]],
        [slice(None), slice(None), [0, 1, 3], [4]],
        [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3]]],
        [slice(None), slice(None), [[0, 1], [2, 3]], [[0]]],
        [slice(None), slice(None), [[5, 6]], [[0, 3], [4, 4]]],
        [slice(None), [0, 2, 3], [1, 3, 4], slice(None)],
        [slice(None), [0], [1, 2, 4], slice(None)],
        [slice(None), [0, 1, 3], [4], slice(None)],
        [slice(None), [[0, 1], [3, 4]], [[2, 3], [0, 1]], slice(None)],
        [slice(None), [[0, 1], [3, 4]], [[2, 3]], slice(None)],
        [slice(None), [[0, 1], [3, 2]], [[0]], slice(None)],
        [slice(None), [[2, 1]], [[0, 3], [6, 4]], slice(None)],
        [slice(None), [[2]], [[0, 3], [4, 2]], slice(None)],
        [[0, 1, 2], [1, 3, 4], slice(None), slice(None)],
        [[0], [1, 2, 4], slice(None), slice(None)],
        [[0, 1, 2], [4], slice(None), slice(None)],
        [[[0, 1], [0, 2]], [[2, 4], [1, 5]], slice(None), slice(None)],
        [[[0, 1], [1, 2]], [[2, 0]], slice(None), slice(None)],
        [[[2, 2]], [[0, 3], [4, 5]], slice(None), slice(None)],
        [[[2]], [[0, 3], [4, 5]], slice(None), slice(None)],
        [slice(None), [3, 4, 6], [0, 2, 3], [1, 3, 4]],
        [slice(None), [2, 3, 4], [1, 3, 4], [4]],
        [slice(None), [0, 1, 3], [4], [1, 3, 4]],
        [slice(None), [6], [0, 2, 3], [1, 3, 4]],
        [slice(None), [2, 3, 5], [3], [4]],
        [slice(None), [0], [4], [1, 3, 4]],
        [slice(None), [6], [0, 2, 3], [1]],
        [slice(None), [[0, 3], [3, 6]], [[0, 1], [1, 3]], [[5, 3], [1, 2]]],
        [[2, 2, 1], [0, 2, 3], [1, 3, 4], slice(None)],
        [[2, 0, 1], [1, 2, 3], [4], slice(None)],
        [[0, 1, 2], [4], [1, 3, 4], slice(None)],
        [[0], [0, 2, 3], [1, 3, 4], slice(None)],
        [[0, 2, 1], [3], [4], slice(None)],
        [[0], [4], [1, 3, 4], slice(None)],
        [[1], [0, 2, 3], [1], slice(None)],
        [[[1, 2], [1, 2]], [[0, 1], [2, 3]], [[2, 3], [3, 5]], slice(None)],
        # less dim, ellipsis
        [Ellipsis, [0, 3, 4]],
        [Ellipsis, slice(None), [0, 3, 4]],
        [Ellipsis, slice(None), slice(None), [0, 3, 4]],
        [slice(None), Ellipsis, [0, 3, 4]],
        [slice(None), slice(None), Ellipsis, [0, 3, 4]],
        [slice(None), [0, 2, 3], [1, 3, 4]],
        [slice(None), [0, 2, 3], [1, 3, 4], Ellipsis],
        [Ellipsis, [0, 2, 3], [1, 3, 4], slice(None)],
        [[0], [1, 2, 4]],
        [[0], [1, 2, 4], slice(None)],
        [[0], [1, 2, 4], Ellipsis],
        [[0], [1, 2, 4], Ellipsis, slice(None)],
        [[1],],
        [[0, 2, 1], [3], [4]],
        [[0, 2, 1], [3], [4], slice(None)],
        [[0, 2, 1], [3], [4], Ellipsis],
        [Ellipsis, [0, 2, 1], [3], [4]],
    ]

    for indexer in indices_to_test:
        assert_get_eq(reference, indexer)
        assert_set_eq(reference, indexer, 1333)
        assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
    indices_to_test += [
        [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3], [3, 0]]],
        [slice(None), slice(None), [[2]], [[0, 3], [4, 4]]],
    ]
    for indexer in indices_to_test:
        assert_get_eq(reference, indexer)
        assert_set_eq(reference, indexer, 1333)
        if device != "cpu":
            assert_backward_eq(reference, indexer)


def _test_single_int(test_case, device):
    v = flow.randn(5, 7, 3, device=device)
    test_case.assertEqual(v[4].shape, (7, 3))


def _test_multiple_int(test_case, device):
    v = flow.randn(5, 7, 3, device=device)
    test_case.assertEqual(v[4].shape, (7, 3))
    test_case.assertEqual(v[4, :, 1].shape, (7,))


def _test_none(test_case, device):
    v = flow.randn(5, 7, 3, device=device)
    test_case.assertEqual(v[None].shape, (1, 5, 7, 3))
    test_case.assertEqual(v[:, None].shape, (5, 1, 7, 3))
    test_case.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
    test_case.assertEqual(v[..., None].shape, (5, 7, 3, 1))


def _test_step(test_case, device):
    v = flow.arange(10, device=device)
    _assert_tensor_equal(test_case, v[::1], v)
    test_case.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
    test_case.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
    test_case.assertEqual(v[::11].tolist(), [0])
    test_case.assertEqual(v[1:6:2].tolist(), [1, 3, 5])


def _test_step_assignment(test_case, device):
    v = flow.zeros(4, 4, device=device)
    v[0, 1::2] = flow.tensor([3.0, 4.0], device=device)
    test_case.assertEqual(v[0].tolist(), [0.0, 3.0, 0.0, 4.0])
    test_case.assertEqual(v[1:].sum(), 0)


def _test_bool_indices(test_case, device):
    v = flow.randn(5, 7, 3, device=device)
    boolIndices = flow.tensor(
        [True, False, True, True, False], dtype=flow.bool, device=device
    )
    test_case.assertEqual(v[boolIndices].shape, (3, 7, 3))
    _assert_tensor_equal(test_case, v[boolIndices], flow.stack([v[0], v[2], v[3]]))

    v = flow.tensor([True, False, True], dtype=flow.bool, device=device)
    boolIndices = flow.tensor([True, False, False], dtype=flow.bool, device=device)
    uint8Indices = flow.tensor([1, 0, 0], dtype=flow.uint8, device=device)
    test_case.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
    test_case.assertEqual(v[boolIndices], v[uint8Indices])
    test_case.assertEqual(
        v[boolIndices], flow.tensor([True], dtype=flow.bool, device=device)
    )


def _test_multiple_bool_indices(test_case, device):
    v = flow.randn(5, 7, 3, device=device)
    # NOTE: these broadcast together and are transposed to the first dim
    mask1 = flow.tensor([1, 0, 1, 1, 0], dtype=flow.bool, device=device)
    mask2 = flow.tensor([1, 1, 1], dtype=flow.bool, device=device)
    test_case.assertEqual(v[mask1, :, mask2].shape, (3, 7))


def _test_int_indices(test_case, device):
    v = flow.randn(5, 7, 3, device=device)
    test_case.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
    test_case.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
    test_case.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))


def _test_int_indices2d(test_case, device):
    x = flow.arange(0, 12, device=device).view(4, 3)
    rows = flow.tensor([[0, 0], [3, 3]], device=device)
    columns = flow.tensor([[0, 2], [0, 2]], device=device)
    test_case.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])


def _test_int_indices_broadcast(test_case, device):
    x = flow.arange(0, 12, device=device).view(4, 3)
    rows = flow.tensor([0, 3], device=device)
    columns = flow.tensor([0, 2], device=device)
    result = x[rows[:, None], columns]
    test_case.assertEqual(result.tolist(), [[0, 2], [9, 11]])


def _test_empty_index(test_case, device):
    x = flow.arange(0, 12, device=device).view(4, 3)
    idx = flow.tensor([], dtype=flow.long, device=device)
    test_case.assertEqual(x[idx].numel(), 0)

    # empty assignment should have no effect but not throw an exception
    y = x.clone()
    y[idx] = -1
    _assert_tensor_equal(test_case, x, y)

    mask = flow.zeros(4, 3, device=device).to(flow.bool)
    y[mask] = -1
    _assert_tensor_equal(test_case, x, y)


def _test_empty_ndim_index(test_case, device):
    x = flow.randn(5, device=device)
    _assert_tensor_equal(
        test_case,
        flow.empty(0, 2, device=device),
        x[flow.empty(0, 2, dtype=flow.int64, device=device)],
    )

    x = flow.randn(2, 3, 4, 5, device=device)
    _assert_tensor_equal(
        test_case,
        flow.empty(2, 0, 6, 4, 5, device=device),
        x[:, flow.empty(0, 6, dtype=flow.int64, device=device)],
    )

    x = flow.empty(10, 0, device=device)
    test_case.assertEqual(x[[1, 2]].shape, (2, 0))
    test_case.assertEqual(x[[], []].shape, (0,))
    test_case.assertEqual(x[[[]]].shape, (0, 0))
    test_case.assertEqual(x[[[[]]]].shape, (1, 0, 0))
    test_case.assertEqual(x[[1], []].shape, (0,))
    test_case.assertEqual(x[[], [2]].shape, (0,))
    with test_case.assertRaisesRegex(IndexError, "for dimension with size 0"):
        x[:, [0, 1]]


def _test_empty_ndim_index_bool(test_case, device):
    x = flow.randn(5, device=device)
    test_case.assertRaises(
        IndexError, lambda: x[flow.empty(0, 2, dtype=flow.uint8, device=device)]
    )


def _test_empty_slice(test_case, device):
    x = flow.randn(2, 3, 4, 5, device=device)
    y = x[:, :, :, 1]
    z = y[:, 1:1, :]
    test_case.assertEqual((2, 0, 4), z.shape)
    # this isn't technically necessary, but matches NumPy stride calculations.
    test_case.assertEqual((60, 20, 5), z.stride())
    test_case.assertTrue(z.is_contiguous())


def _test_index_getitem_copy_bools_slices(test_case, device):
    true = flow.tensor(1, dtype=flow.uint8, device=device)
    false = flow.tensor(0, dtype=flow.uint8, device=device)

    tensors = [flow.randn(2, 3, device=device), flow.tensor([1.0], device=device)]

    # TODO: compare tensor_storage after exporting the inferface
    for a in tensors:
        #  test_case.assertNotEqual(a.data_ptr(), a[True].data_ptr())
        _assert_tensor_equal(test_case, flow.empty(0, *a.shape), a[False])
        #  test_case.assertNotEqual(a.data_ptr(), a[true].data_ptr())
        _assert_tensor_equal(test_case, flow.empty(0, *a.shape), a[false])
        #  test_case.assertEqual(a.data_ptr(), a[None].data_ptr())
        #  test_case.assertEqual(a.data_ptr(), a[...].data_ptr())


def _test_setitem_scalars(test_case, device):
    zero = flow.tensor(0, dtype=flow.int64)

    # non-scalar indexed with scalars
    a = flow.randn(2, 3, device=device)
    a_set_with_number = a.clone()
    a_set_with_scalar = a.clone()
    b = flow.randn(3, device=device)

    a_set_with_number[0] = b
    a_set_with_scalar[zero] = b
    _assert_tensor_equal(test_case, a_set_with_number, a_set_with_scalar)
    a[1, zero] = 7.7
    value = a[1, 0].numpy()
    test_case.assertEqual(np.array(7.7, dtype=value.dtype), value)

    np_x = np.random.rand(2, 3)
    np_x[0, 0] = 1.0
    x = flow.tensor(np_x)
    x[0, 0] = 1.0
    test_case.assertEqual(x.numpy().all(), np_x.all())

    # scalar indexed with scalars
    r = flow.tensor(1.0).to(device)
    with test_case.assertRaises(IndexError):
        r[:] = 8.8
    with test_case.assertRaises(IndexError):
        r[zero] = 8.8
    r[...] = 9.9
    test_case.assertEqual(r, 9.9)

    # scalar indexed with oneflow.Size([1])
    np_x = np.random.rand(2, 3)
    np_x[0, 0] = np.ones(1)
    x = flow.tensor(np_x)
    x[0, 0] = flow.ones(1).to(flow.float64)
    test_case.assertEqual(x.numpy().all(), np_x.all())


def _test_basic_advanced_combined(test_case, device):
    x = flow.arange(0, 12, device=device).view(4, 3)
    _assert_tensor_equal(test_case, x[1:2, 1:3], x[1:2, [1, 2]])
    test_case.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

    # Check that it is a copy
    unmodified = x.clone()
    x[1:2, [1, 2]].zero_()
    _assert_tensor_equal(test_case, x, unmodified)

    # But assignment should modify the original
    unmodified = x.clone()
    x[1:2, [1, 2]] = 0
    test_case.assertFalse(np.array_equal(x.numpy(), unmodified.numpy()))


def _test_ellipsis_tensor(test_case, device):
    x = flow.arange(0, 9, device=device).view(3, 3)
    idx = flow.tensor([0, 2], device=device)
    test_case.assertEqual(x[..., idx].tolist(), [[0, 2], [3, 5], [6, 8]])
    test_case.assertEqual(x[idx, ...].tolist(), [[0, 1, 2], [6, 7, 8]])

    # Test scalar ellipsis getitem
    y = flow.tensor(1.0).to(device)
    x_scalar = flow.tensor(9.9)
    y = x_scalar[...]
    test_case.assertEqual(y, 9.9)


@flow.unittest.skip_unless_1n1d()
class TestIndexing(flow.unittest.TestCase):
    def test_slice(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgDict(arg_dict):
            dtype_list = [flow.float32, flow.float16]
            from oneflow import sysconfig

            if not sysconfig.get_cuda_version() < 11000:
                dtype_list.append(flow.bfloat16)

            for dtype in dtype_list:
                _test_basic_slice(test_case, **arg, dtype=dtype)
                _test_advanced_indexing(test_case, **arg, dtype=dtype)
                _test_combined_indexing(test_case, **arg, dtype=dtype)
            _test_single_int(test_case, **arg)
            _test_multiple_int(test_case, **arg)
            _test_none(test_case, **arg)
            _test_step(test_case, **arg)
            _test_step_assignment(test_case, **arg)
            _test_bool_indices(test_case, **arg)
            _test_multiple_bool_indices(test_case, **arg)
            _test_int_indices(test_case, **arg)
            _test_int_indices2d(test_case, **arg)
            _test_int_indices_broadcast(test_case, **arg)
            _test_empty_index(test_case, **arg)
            _test_empty_ndim_index(test_case, **arg)
            _test_empty_ndim_index_bool(test_case, **arg)
            _test_empty_slice(test_case, **arg)
            _test_index_getitem_copy_bools_slices(test_case, **arg)
            _test_setitem_scalars(test_case, **arg)
            _test_basic_advanced_combined(test_case, **arg)
            _test_ellipsis_tensor(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
