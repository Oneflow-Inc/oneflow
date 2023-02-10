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
import itertools
import numpy as np
from itertools import repeat

import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.parallel import comm
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

TEST_CUDA = flow.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and flow.cuda.device_count() >= 2
"""
This test unit is referenced from pytorch.
/pytorch/test/test_cuda.py
"""


@flow.unittest.skip_unless_1n1d()
class TestDp(flow.unittest.TestCase):
    def _test_scatter(self, input, chunk_sizes=None, dim=0):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        if chunk_sizes is None:
            ref_chunk_sizes = tuple(repeat(input.size(dim) // 2, 2))
        else:
            ref_chunk_sizes = chunk_sizes

        # test regular
        result = comm.scatter(input, (0, 1), chunk_sizes, dim)
        self.assertEqual(len(result), 2)
        chunk_start = 0
        for i, r in enumerate(result):
            chunk_end = chunk_start + ref_chunk_sizes[i]
            index = [slice(None, None) for _ in range(input.dim())]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertTrue(
                flow.allclose(r.cpu(), input[tuple(index)].cpu(), atol=0, rtol=0)
            )
            # self.assertEqual(r, input[tuple(index)], atol=0, rtol=0)
            chunk_start = chunk_end
            # if r.device == input.device:
            #     self.assertEqual(r.data_ptr(), input.data_ptr())  # for target @ same device, a view should be returned

        # test out
        out = [flow.empty_like(t) for t in result]
        result = comm.scatter(input, dim=dim, out=out)
        self.assertEqual(len(result), 2)
        chunk_start = 0
        for i, r in enumerate(result):
            self.assertIs(r, out[i])
            chunk_end = chunk_start + ref_chunk_sizes[i]
            index = [slice(None, None) for _ in range(input.dim())]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertTrue(
                flow.allclose(r.cpu(), input[tuple(index)].cpu(), atol=0, rtol=0)
            )
            # self.assertEqual(r, input[tuple(index)], atol=0, rtol=0)
            chunk_start = chunk_end

        # test error msg
        if chunk_sizes is not None:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected devices and chunk_sizes to be of same length"
            ):
                comm.scatter(
                    input,
                    [0 for _ in range(len(chunk_sizes) + 1)],
                    dim=dim,
                    chunk_sizes=chunk_sizes,
                )
        with self.assertRaisesRegex(RuntimeError, r"'devices' must not be specified"):
            comm.scatter(input, (0, 1), dim=dim, out=out)
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one device to scatter to"
        ):
            comm.scatter(input, (), dim=dim)
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one output tensor to scatter to"
        ):
            comm.scatter(input, dim=dim, out=[])
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to be CUDA tensors, but output tensor at index 0",
        ):
            comm.scatter(input, dim=dim, out=([out[0].cpu()] + out[1:]))
        with self.assertRaisesRegex(
            RuntimeError, r"Output tensor at index 0 has incorrect shape"
        ):
            comm.scatter(input, dim=dim, out=([out[0].unsqueeze(0)] + out[1:]))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Total size for output tensors along scatter dim does not match",
        ):
            index = [slice(None, None) for _ in range(input.dim())]
            index[dim] = slice(1, None)
            comm.scatter(input, dim=dim, out=([out[0][tuple(index)]] + out[1:]))

    def test_scatter_cpu(self):
        self._test_scatter(flow.randn(4, 4), dim=0)

    def test_scatter_cpu_dim(self):
        self._test_scatter(flow.randn(4, 4), dim=1)

    def test_scatter_cpu_neg_dim(self):
        self._test_scatter(flow.randn(4, 4), dim=-2)

    def test_scatter_cpu_sizes(self):
        self._test_scatter(flow.randn(6, 4), chunk_sizes=(2, 4))

    def test_scatter_gpu(self):
        self._test_scatter(flow.randn(4, 4).cuda(), dim=0)

    def test_scatter_gpu_dim(self):
        self._test_scatter(flow.randn(4, 4).cuda(), dim=1)

    def test_scatter_gpu_neg_dim(self):
        self._test_scatter(flow.randn(4, 4).cuda(), dim=-2)

    def test_scatter_gpu_sizes(self):
        self._test_scatter(flow.randn(6, 4).cuda(), chunk_sizes=(2, 4))

    def _test_broadcast(self, input):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        # test regular
        results = comm.broadcast(input, (0, 1))
        for i, t in enumerate(results):
            self.assertEqual(t.get_device(), i)
            self.assertTrue(flow.allclose(t.cpu(), input.cpu(), atol=0, rtol=0))
            if (
                input.is_cuda and input.get_device() == i
            ):  # test not copying on same device
                self.assertEqual(id(t), id(input))
                # self.assertEqual(t.data_ptr(), input.data_ptr())
        # test out=
        for inplace in [True, False]:
            if inplace:
                outputs = [
                    flow.empty_like(input, device=flow.device("cuda", 0)),
                    flow.empty_like(input, device=flow.device("cuda", 1)),
                ]
            else:
                outputs = [
                    input.cuda(0),
                    flow.empty_like(input, device=flow.device("cuda", 1)),
                ]
            results = comm.broadcast(input, out=outputs)
            for r, o in zip(results, outputs):
                self.assertIs(r, o)
            for i, t in enumerate(results):
                self.assertEqual(t.get_device(), i)
                self.assertTrue(flow.allclose(t.cpu(), input.cpu(), atol=0, rtol=0))
                # self.assertEqual(t, input)
        # test error msg
        with self.assertRaisesRegex(
            RuntimeError, r"Exactly one of 'devices' and 'out'"
        ):
            comm.broadcast(input, (0, 1), out=outputs)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to be CUDA tensors, but output tensor at index 1",
        ):
            comm.broadcast(input, out=[input.cuda(0), input.cpu()])
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to have same shape as the source at index 1",
        ):
            comm.broadcast(input, out=[input.cuda(0), input.cuda(1).unsqueeze(0)])

    def test_broadcast_cpu(self):
        self._test_broadcast(flow.randn(5, 5))

    def test_broadcast_gpu(self):
        self._test_broadcast(flow.randn(5, 5).cuda())

    def _test_broadcast_coalesced(self, tensors, buffer_size):
        b_tensors = [comm.broadcast(t, (0, 1)) for t in tensors]
        for (_, bt), t in zip(b_tensors, tensors):
            self.assertEqual(bt.get_device(), 1)
            self.assertTrue(flow.allclose(bt.cpu(), t.cpu(), atol=0, rtol=0))
            # self.assertEqual(bt, t)
            self.assertIsInstance(bt, type(t))

        bc_tensors = comm.broadcast_coalesced(tensors, (0, 1), buffer_size=buffer_size)
        bc_tensors_t = list(zip(*bc_tensors))
        for x, y in zip(bc_tensors_t, b_tensors):
            for tensor_x, tensor_y in zip(x, y):
                self.assertTrue(flow.allclose(tensor_x, tensor_y, atol=0, rtol=0))
        # self.assertEqual(b_tensors, bc_tensors_t)
        for (_, bt), (_, bct) in zip(b_tensors, bc_tensors_t):
            self.assertEqual(bt.get_device(), bct.get_device())
            self.assertIsInstance(bct, type(bt))

        # check that tensors on device[0] are returned as-is
        for out_tensors in (b_tensors, bc_tensors_t):
            for inp_t, (out_t, _) in zip(tensors, out_tensors):
                self.assertIs(inp_t, out_t)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # Note: fails sometimes on the CI, passes on dual gfx906
    def test_broadcast_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            flow.randn(numel).long().cuda(),
            flow.randn(numel).cuda(),
            flow.randn(numel).long().cuda(),
            flow.randn(numel).long().cuda(),
            flow.randn(numel * 2).int().cuda(),  # int is 2x shorter
            flow.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_broadcast_coalesced_empty_tensors(self):
        tensors = [
            flow.tensor([]).byte().cuda(),
            flow.randn(5).cuda(),
            flow.randn(5).double().cuda(),
        ]
        self._test_broadcast_coalesced(tensors, 256)

    def _test_gather(self, dim):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        x = flow.randn(2, 5, device=flow.device("cuda", 0))
        y = flow.randn(2, 5, device=flow.device("cuda", 1))
        expected_size = list(x.size())
        expected_size[dim] += y.size(dim)
        expected_size = flow.Size(expected_size)

        destinations = [None, flow.device("cuda:0"), flow.device("cpu")]
        if flow.cuda.device_count() > 2:
            destinations.append(flow.device("cuda:2"))
        # with flow.cuda.device(flow.device('cuda', 1)):
        for destination in destinations:
            if destination is None:
                expected_device = flow.device("cuda", 1)
            else:
                expected_device = destination
            for use_out in [True, False]:
                if use_out:
                    out = flow.empty(expected_size, device=expected_device)
                    result = comm.gather((x, y), dim, out=out)
                    self.assertIs(out, result)
                else:
                    result = comm.gather((x, y), dim, destination=expected_device)

                self.assertEqual(result.device, expected_device)
                self.assertEqual(result.size(), expected_size)

                index = [slice(None, None), slice(None, None)]
                index[dim] = slice(0, x.size(dim))
                self.assertTrue(
                    flow.allclose(result[tuple(index)].cpu(), x.cpu(), atol=0, rtol=0)
                )
                # self.assertEqual(result[tuple(index)], x)
                index[dim] = slice(x.size(dim), x.size(dim) + y.size(dim))
                self.assertTrue(
                    flow.allclose(result[tuple(index)].cpu(), y.cpu(), atol=0, rtol=0)
                )
                # self.assertEqual(result[tuple(index)], y)

        # test error msg
        with self.assertRaisesRegex(
            RuntimeError, r"'destination' must not be specified"
        ):
            comm.gather(
                (x, y),
                dim,
                destination="cpu",
                out=torch.empty(expected_size, device="cpu"),
            )
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one tensor to gather from"
        ):
            comm.gather(())
        with self.assertRaisesRegex(
            RuntimeError, r"Expected all input tensors to be CUDA tensors, "
        ):
            comm.gather((x.cpu(), y))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all input tensors to have the same number of dimensions",
        ):
            comm.gather((x, y.unsqueeze(0)))
        with self.assertRaisesRegex(
            RuntimeError, r"Input tensor at index 1 has invalid shape"
        ):
            if dim in [0, -2]:
                comm.gather((x, y[:, 1:]), dim=dim)
            elif dim in [1, -1]:
                comm.gather((x, y[1:, :]), dim=dim)

    def test_gather(self):
        self._test_gather(0)

    def test_gather_dim(self):
        self._test_gather(1)

    def test_gather_neg_dim(self):
        self._test_gather(-1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add(self):
        x = flow.randn(5, 5)
        y = flow.randn(5, 5)
        x_cuda = x.cuda(0)
        y_cuda = y.cuda(1)
        result = comm.reduce_add((x_cuda, y_cuda))
        self.assertEqual(result.get_device(), 0)
        self.assertTrue(flow.allclose(result.cpu(), x + y, atol=0, rtol=0))

    def _test_reduce_add_coalesced(self, tensors, buffer_size):
        dup_tensors = [tensors, [t.cuda(1) for t in tensors]]

        r_tensors = [comm.reduce_add(t) for t in zip(*dup_tensors)]
        for r, t in zip(r_tensors, tensors):
            self.assertEqual(r.device, t.device)
            self.assertEqual(r.dtype, t.dtype)
            self.assertTrue(flow.allclose(r, t * 2, atol=0, rtol=0))

        rc_tensors = comm.reduce_add_coalesced(dup_tensors, buffer_size=buffer_size)

        for r, rc in zip(r_tensors, rc_tensors):
            self.assertEqual(r.device, rc.device)
            self.assertEqual(r.dtype, rc.dtype)
            self.assertTrue(flow.allclose(r, rc, atol=0, rtol=0))

        # Since we have both cuda:0 and cuda:1 inputs, the outputs must be new.
        # We can check that they have different version counters.
        # NOTE [ Version Counter in comm.*_coalesced ]
        # versions = [t._version for t in rc_tensors]
        # for old_version, t in zip(versions, rc_tensors):
        #     self.assertEqual(t._version, old_version)
        #     t.zero_()
        #     self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            flow.randn(numel).long().cuda(),
            flow.randn(numel).cuda(),
            flow.randn(numel).long().cuda(),
            flow.randn(numel).long().cuda(),
            flow.randn(numel * 2).int().cuda(),  # int is 2x shorter
            flow.randn(numel).cuda(),
        ]
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)


if __name__ == "__main__":
    unittest.main()
