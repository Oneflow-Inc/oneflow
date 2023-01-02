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
import numpy as np

import oneflow as flow
import oneflow.unittest
import torch

from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


def _test_global_expand(
    test_case,
    input_shape,
    expand_shape,
    device="cuda",
    sbp=flow.sbp.broadcast,
    verbose=False,
):
    # random input
    input = np.random.randn(*input_shape)
    if isinstance(input, np.ndarray):
        input = input.astype(np.float32)

    # torch computation
    torch_x = torch.tensor(input, requires_grad=True)
    torch_y = torch_x.expand(*expand_shape)
    torch_y.sum().backward()

    # oneflow computation
    placement = flow.placement(device, np.array(range(flow.env.get_world_size())))
    x = flow.tensor(input, requires_grad=True)
    global_x = x.to_global(placement=placement, sbp=flow.sbp.broadcast)
    if global_x.sbp != sbp:
        global_x = global_x.to_global(sbp=sbp, grad_sbp=flow.sbp.broadcast)
    y = global_x.expand(*expand_shape)
    y.sum().backward()

    y_b = y.to_global(sbp=flow.sbp.broadcast)

    if flow.env.get_rank() == 0:
        out_a = y_b.to_local().numpy()
        out_b = torch_y.detach().cpu().numpy()
        grad_a = x.grad.numpy()
        grad_b = torch_x.grad.cpu().numpy()

        if verbose:
            print("")
            print(f"{'=' * 10} {input_shape} -> {expand_shape} {'=' * 10}")
            print(f"{'=' * 10} {device}, {sbp} {'=' * 10}")
            print(f"{'-' * 20} compare out {'-' * 20}")
            print(out_a)
            print("*" * 20)
            print(out_b)
            print("")
            print(f"{'-' * 20} compare grad {'-' * 20}")
            print(grad_a)
            print("*" * 20)
            print(grad_b)

        test_case.assertTrue(np.array_equal(out_a, out_b))
        test_case.assertTrue(np.array_equal(grad_a, grad_b))


@flow.unittest.skip_unless_1n2d()
class ExpandGlobalTestCase(oneflow.unittest.TestCase):
    def test_global_expand(test_case):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["sbp"] = [flow.sbp.split(0), flow.sbp.broadcast()]
        arg_dict["shapes"] = [
            ((2, 2), (2, 2, 2)),
            ((2, 1, 3), (2, 1, -1, -1, -1)),
            ((2, 1, 3), (1, 2, -1, -1, -1)),
            ((2, 1, 3), (2, 1, -1, 2, 3)),
            ((2, 1, 3), (1, 2, 2, 2, -1)),
        ]
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _test_global_expand(test_case, input_shape, expand_shape, **kwargs)

    def test_split_expand(test_case):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cuda"]
        arg_dict["sbp"] = [flow.sbp.split(0)]
        arg_dict["shapes"] = [
            ((2,), (1, 2)),
            ((2,), (2, 2)),
        ]
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _test_global_expand(test_case, input_shape, expand_shape, **kwargs)

    def test_broadcast_scalar_expand(test_case):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["sbp"] = [flow.sbp.broadcast()]
        arg_dict["shapes"] = [
            ((), (1,)),
            ((), (2,)),
            ((), (1, 1)),
            ((), (1, 2)),
            ((), (2, 1)),
            ((), (2, 2)),
            ((), (2, 1, 2)),
        ]
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _test_global_expand(test_case, input_shape, expand_shape, **kwargs)


if __name__ == "__main__":
    unittest.main()

# ONEFLOW_TEST_DEVICE_NUM=2 python3 -m oneflow.distributed.launch --nproc_per_node 2 test_global_expand_op.py
