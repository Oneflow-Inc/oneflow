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
import os
import numpy as np
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

import torch


def _test_expand_new_dims_broadcast(test_case, device):
    input_shape = (1, 4, 1, 1)
    expand_dim = [2, 1, 2, 4, 2, 1]

    input_nd = np.random.random(size=input_shape).astype(np.float32)
    torch_in = torch.tensor(input_nd, requires_grad=True)
    torch_out = torch_in.expand(*expand_dim)
    torch_out.sum().backward()

    of_input = flow.tensor(input_nd, dtype=flow.float32, requires_grad=True)
    global_of_input = of_input.to_global(
        placement=flow.placement(device, ranks=[0, 1]), sbp=flow.sbp.broadcast,
    )
    of_out = global_of_input.expand(*expand_dim)
    of_out.sum().backward()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(of_out.to_local().numpy(), torch_out.detach().cpu().numpy())
        )
        test_case.assertTrue(
            np.array_equal(of_input.grad.numpy(), torch_in.grad.cpu().numpy())
        )


def _test_expand_same_dim_broadcast(test_case, device):
    input_shape = (4, 1, 2, 1)
    expand_dim = [4, 1, 2, 1]

    input_nd = np.random.random(size=input_shape).astype(np.float32)
    torch_in = torch.tensor(input_nd, requires_grad=True)
    torch_out = torch_in.expand(*expand_dim)
    torch_out.sum().backward()

    of_input = flow.tensor(input_nd, dtype=flow.float32, requires_grad=True)
    global_of_input = of_input.to_global(
        placement=flow.placement(device, ranks=[0, 1]), sbp=flow.sbp.broadcast,
    )

    of_out = global_of_input.expand(*expand_dim)
    loss = of_out.sum()
    loss.backward()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(of_out.to_local().numpy(), torch_out.detach().cpu().numpy())
        )
        test_case.assertTrue(
            np.array_equal(of_input.grad.numpy(), torch_in.grad.cpu().numpy())
        )


def _test_expand_same_dim_negative_broadcast(test_case, device):
    input_shape = (2, 1, 4, 1)
    expand_dim = [2, -1, 4, 4]

    input_nd = np.random.random(size=input_shape).astype(np.float32)
    torch_in = torch.tensor(input_nd, requires_grad=True)
    torch_out = torch_in.expand(*expand_dim)
    torch_out.sum().backward()

    of_input = flow.tensor(input_nd, dtype=flow.float32, requires_grad=True)
    global_of_input = of_input.to_global(
        placement=flow.placement(device, ranks=[0, 1]), sbp=flow.sbp.broadcast,
    )

    of_out = global_of_input.expand(*expand_dim)
    loss = of_out.sum()
    loss.backward()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(of_out.to_local().numpy(), torch_out.detach().cpu().numpy())
        )
        test_case.assertTrue(
            np.array_equal(of_input.grad.numpy(), torch_in.grad.cpu().numpy())
        )


def _test_expand_new_dims_split(test_case, device):
    input_shape = (4, 1, 2, 1)
    expand_dim = [2, 1, 4, 1, 2, 1]

    input_nd = np.random.random(size=input_shape).astype(np.float32)
    torch_in = torch.tensor(input_nd, requires_grad=True)
    torch_out = torch_in.expand(*expand_dim)
    torch_out.sum().backward()

    of_input = flow.tensor(input_nd, dtype=flow.float32, requires_grad=True)
    global_of_input = of_input.to_global(
        placement=flow.placement(device, ranks=[0, 1]), sbp=flow.sbp.broadcast,
    )
    global_of_input = global_of_input.to_global(sbp=flow.sbp.split(0))

    of_out = global_of_input.expand(*expand_dim)
    loss = of_out.sum()
    loss.backward()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                of_out.to_local().numpy(),
                torch_out.detach().cpu().numpy()[:, :, 0:2, :, :, :],
            )
        )
        test_case.assertTrue(
            np.array_equal(
                of_input.grad.numpy(), torch_in.grad.cpu().numpy()[0:2, :, :, :],
            )
        )


def _test_expand_same_dim_split(test_case, device):
    input_shape = (4, 1, 2, 1)
    expand_dim = [4, 1, 2, 4]

    input_nd = np.random.random(size=input_shape).astype(np.float32)
    torch_in = torch.tensor(input_nd, requires_grad=True)
    torch_out = torch_in.expand(*expand_dim)
    torch_out.sum().backward()

    of_input = flow.tensor(input_nd, dtype=flow.float32, requires_grad=True)
    global_of_input = of_input.to_global(
        placement=flow.placement(device, ranks=[0, 1]), sbp=flow.sbp.broadcast,
    )
    global_of_input = global_of_input.to_global(sbp=flow.sbp.split(0))

    of_out = of_input.expand(*expand_dim)
    loss = of_out.sum()
    loss.backward()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                of_out.to_local().numpy(),
                torch_out.detach().cpu().numpy()[0:2, :, :, :],
            )
        )
        test_case.assertTrue(
            np.array_equal(
                of_input.grad.numpy(), torch_in.grad.cpu().numpy()[0:2, :, :, :],
            )
        )


def _test_expand_same_dim_negative_split(test_case, device):
    input_shape = (2, 1, 4, 1)
    expand_dim = [2, -1, 4, 4]

    input_nd = np.random.random(size=input_shape).astype(np.float32)
    torch_in = torch.tensor(input_nd, requires_grad=True)
    torch_out = torch_in.expand(*expand_dim)
    torch_out.sum().backward()

    of_input = flow.tensor(input_nd, dtype=flow.float32, requires_grad=True)
    global_of_input = of_input.to_global(
        placement=flow.placement(device, ranks=[0, 1]), sbp=flow.sbp.broadcast,
    )
    global_of_input = global_of_input.to_global(sbp=flow.sbp.split(2))

    of_out = global_of_input.expand(*expand_dim)
    loss = of_out.sum()
    loss.backward()

    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                of_out.to_local().numpy(),
                torch_out.detach().cpu().numpy()[:, :, 0:2, :],
            )
        )
        test_case.assertTrue(
            np.array_equal(
                of_input.grad.numpy(), torch_in.grad.cpu().numpy()[:, :, 0:2, :],
            )
        )


@flow.unittest.skip_unless_1n2d()
class ExpandGlobalTestCase(oneflow.unittest.TestCase):
    def test_expand_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_expand_new_dims_broadcast,
            _test_expand_same_dim_broadcast,
            _test_expand_same_dim_negative_broadcast,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    # NOTE(Liang Depeng): Run with the following command can pass the test locally, but will fail when run in ci.
    # ONEFLOW_TEST_DEVICE_NUM=2 python3 -m oneflow.distributed.launch --nproc_per_node 2 test_global_expand_op.py
    @unittest.skipIf(True, "skip for now")
    def test_expand_split(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_expand_new_dims_split,
            _test_expand_same_dim_split,
            _test_expand_same_dim_negative_split,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
