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
import numpy as np
import inspect
import types
import unittest
import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


# y1 = rand_op1(x)
# y2 = rand_op2(x)
# rand_op1 and rand_op2 should have different seed in graph, then lead to different result
def _inspect_rand_op_and_args(rand_op, **kwargs):
    if inspect.isclass(rand_op) and issubclass(rand_op, nn.Module):
        init_method_signature = inspect.signature(rand_op.__init__)

        module_init_args = dict()
        for arg_name in list(init_method_signature.parameters.keys())[1:]:
            if arg_name in kwargs:
                module_init_args[arg_name] = kwargs.pop(arg_name)

        module_instance = rand_op(**module_init_args)
        return module_instance, kwargs

    if isinstance(rand_op, types.BuiltinFunctionType):
        return rand_op, kwargs

    if inspect.isfunction(rand_op):
        return rand_op, kwargs

    raise ValueError(f"invalid rand_op {rand_op}, type: {type(rand_op)}")


def _test_rand_op_unidentical(test_case, rand_op, input=None, **kwargs):
    rand_op1, kwargs1 = _inspect_rand_op_and_args(rand_op, **kwargs)
    rand_op2, kwargs2 = _inspect_rand_op_and_args(rand_op, **kwargs)

    if input is None:
        result1 = rand_op1(**kwargs1)
        result2 = rand_op2(**kwargs2)
    else:
        x1 = input
        x2 = input.clone()
        result1 = rand_op1(x1, **kwargs1)
        result2 = rand_op2(x2, **kwargs2)

    if isinstance(result1, (list, tuple)):
        result1 = result1[0]
    if isinstance(result2, (list, tuple)):
        result2 = result2[0]

    test_case.assertFalse(
        np.allclose(result1.numpy(), result2.numpy()),
        f"\ninput:\n{input}\result1:\n{result1}\result2:\n{result2}",
    )


def _test_global_rand_op_with_split(test_case, rand_op, input=None, **kwargs):
    rand_op, kwargs = _inspect_rand_op_and_args(rand_op, **kwargs)
    ranks = np.array(range(flow.env.get_world_size()))

    if input is None:
        device = kwargs.pop("device", None)
        placement = flow.placement(device, ranks)
        y = rand_op(placement=placement, sbp=flow.sbp.split(0), **kwargs)
    else:
        x = flow.concat([input, input], dim=0)
        placement = flow.placement(input.device.type, ranks)
        # local to broadcast global
        x_broadcast = x.to_global(
            placement=placement, sbp=flow.sbp.broadcast(), copy=True
        )
        x_split = x_broadcast.to_global(sbp=flow.sbp.split(0))
        y = rand_op(x_split, **kwargs)

    if isinstance(y, (list, tuple)):
        y = y[0]

    y_broadcast = y.to_global(placement=placement, sbp=flow.sbp.broadcast())
    half = y_broadcast.shape[0] // 2
    first_half = y_broadcast[0:half]
    second_half = y_broadcast[half:]
    test_case.assertFalse(np.allclose(first_half.numpy(), second_half.numpy()))


def _test_global_rand_op_with_broadcast(test_case, rand_op, input=None, **kwargs):
    rand_op, kwargs = _inspect_rand_op_and_args(rand_op, **kwargs)
    ranks = np.array(range(flow.env.get_world_size()))

    if input is None:
        device = kwargs.pop("device", "cpu")
        placement = flow.placement(device, ranks)
        y = rand_op(placement=placement, sbp=flow.sbp.broadcast(), **kwargs)
    else:
        placement = flow.placement(input.device.type, ranks)
        # local to broadcast global
        x = input.to_global(placement=placement, sbp=flow.sbp.broadcast(), copy=True)
        y = rand_op(x, **kwargs)

    if isinstance(y, (list, tuple)):
        y_local = y[0].to_local()
    else:
        y_local = y.to_local()

    y_all_ranks = y_local.to_global(placement=placement, sbp=flow.sbp.split(0))
    y_allgather = y_all_ranks.to_global(sbp=flow.sbp.broadcast())
    half = y_allgather.shape[0] // 2
    first_half = y_allgather[0:half]
    second_half = y_allgather[half:]
    test_case.assertTrue(np.allclose(first_half.numpy(), second_half.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestRandOpUnidentical(oneflow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op(self):
        for device in ("cpu", "cuda"):
            x = flow.randn(4, 16, device=device)
            _test_rand_op_unidentical(self, nn.Dropout, x, p=0.5)
            _test_rand_op_unidentical(self, flow._C.rrelu, x, training=True)
            _test_rand_op_unidentical(self, nn.init.uniform_, x)
            _test_rand_op_unidentical(self, flow._C.exponential_, x)

            x1 = flow.rand(4, 16, device=device)
            _test_rand_op_unidentical(
                self, flow.multinomial, x1, num_samples=16, replacement=True
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op(self):
        shape = (4, 16)
        for device in ("cpu", "cuda"):
            _test_rand_op_unidentical(self, flow.rand, size=shape, device=device)
            _test_rand_op_unidentical(
                self, flow.normal, mean=0.0, std=1.0, size=shape, device=device
            )
            _test_rand_op_unidentical(
                self, flow.randint, low=0, high=10, size=shape, device=device
            )
            _test_rand_op_unidentical(self, flow.randperm, n=32, device=device)

    def test_bernoulli(self):
        x1 = flow.randn(4, 16)
        _test_rand_op_unidentical(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(4, 16)
        _test_rand_op_unidentical(self, flow.bernoulli, x2)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_mask_like(self):
        x = flow.randn(4, 16, 64).to("cuda")
        _test_rand_op_unidentical(
            self,
            flow._C.fused_scale_tril_softmax_mask_scale,
            x,
            p=0.1,
            diagonal=2,
            tril_scale_value=-1000,
        )


@flow.unittest.skip_unless_1n2d()
class TestGlobalRandOp(oneflow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op_with_split(self):
        for device in ("cpu", "cuda"):
            x = flow.randn(2, 4, device=device)
            _test_global_rand_op_with_split(self, nn.Dropout, x, p=0.5)
            _test_global_rand_op_with_split(self, flow._C.rrelu, x, training=True)
            _test_global_rand_op_with_split(self, nn.init.uniform_, x)
            _test_global_rand_op_with_split(self, flow._C.exponential_, x)

            x1 = flow.rand(2, 8, device=device)
            _test_global_rand_op_with_split(
                self, flow.multinomial, x1, num_samples=8, replacement=True
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op_with_broadcast(self):
        for device in ("cpu", "cuda"):
            x = flow.randn(2, 4, device=device)
            _test_global_rand_op_with_broadcast(self, nn.Dropout, x, p=0.5)
            _test_global_rand_op_with_broadcast(self, flow._C.rrelu, x, training=True)
            _test_global_rand_op_with_broadcast(self, nn.init.uniform_, x)
            _test_global_rand_op_with_broadcast(self, flow._C.exponential_, x)

            x1 = flow.rand(2, 8, device=device)
            _test_global_rand_op_with_broadcast(
                self, flow.multinomial, x1, num_samples=8, replacement=True
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op_with_split(self):
        shape = (4, 4)
        for device in ("cpu", "cuda"):
            _test_global_rand_op_with_split(self, flow.rand, size=shape, device=device)
            _test_global_rand_op_with_split(
                self, flow.normal, mean=0.0, std=1.0, size=shape, device=device
            )
            _test_global_rand_op_with_split(
                self, flow.randint, low=0, high=10, size=shape, device=device
            )
            _test_global_rand_op_with_split(self, flow.randperm, n=32, device=device)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op_with_broadcast(self):
        shape = (4, 4)
        for device in ("cpu", "cuda"):
            _test_global_rand_op_with_broadcast(
                self, flow.rand, size=shape, device=device
            )
            _test_global_rand_op_with_broadcast(
                self, flow.normal, mean=0.0, std=1.0, size=shape, device=device
            )
            _test_global_rand_op_with_broadcast(
                self, flow.randint, low=0, high=10, size=shape, device=device
            )
            _test_global_rand_op_with_broadcast(
                self, flow.randperm, n=32, device=device
            )

    def test_bernoulli_with_split(self):
        x1 = flow.randn(2, 8)
        _test_global_rand_op_with_split(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(2, 8)
        _test_global_rand_op_with_split(self, flow.bernoulli, x2)

    def test_bernoulli_with_broadcast(self):
        x1 = flow.randn(2, 8)
        _test_global_rand_op_with_broadcast(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(2, 8)
        _test_global_rand_op_with_broadcast(self, flow.bernoulli, x2)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_mask_like_with_split(self):
        x = flow.randn(2, 16, 64).to("cuda")
        _test_global_rand_op_with_split(
            self,
            flow._C.fused_scale_tril_softmax_mask_scale,
            x,
            p=0.1,
            diagonal=0,
            tril_scale_value=-1000,
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_mask_like_with_broadcast(self):
        x = flow.randn(2, 16, 64).to("cuda")
        _test_global_rand_op_with_broadcast(
            self,
            flow._C.fused_scale_tril_softmax_mask_scale,
            x,
            p=0.2,
            diagonal=1,
            tril_scale_value=-100,
        )


if __name__ == "__main__":
    unittest.main()
