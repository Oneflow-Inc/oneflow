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
import unittest
import inspect
import types
import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


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


# y1 = rand_op1(x)
# y2 = rand_op2(x)
# rand_op1 and rand_op2 should have different seed in graph, lead to different result
def _test_rand_op_in_graph(test_case, rand_op, input=None, **kwargs):
    rand_op1, kwargs1 = _inspect_rand_op_and_args(rand_op, **kwargs)
    rand_op2, kwargs2 = _inspect_rand_op_and_args(rand_op, **kwargs)

    class TestGraphWithoutInput(nn.Graph):
        def __init__(self):
            super().__init__()
            self.rand_op1 = rand_op1
            self.rand_op2 = rand_op2

        def build(self):
            y1 = self.rand_op1(**kwargs1)
            y2 = self.rand_op2(**kwargs2)
            return y1, y2

    class TestGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.rand_op1 = rand_op1
            self.rand_op2 = rand_op2

        def build(self, x):
            x1 = x
            x2 = x.clone()
            y1 = self.rand_op1(x1, **kwargs1)
            y2 = self.rand_op2(x2, **kwargs2)
            return y1, y2

    if input is None:
        graph = TestGraphWithoutInput()
        rand_result1, rand_result2 = graph()
    else:
        graph = TestGraph()
        rand_result1, rand_result2 = graph(input)

    if isinstance(rand_result1, (list, tuple)):
        rand_result1 = rand_result1[0]
    if isinstance(rand_result2, (list, tuple)):
        rand_result2 = rand_result2[0]

    test_case.assertFalse(
        np.allclose(rand_result1.numpy(), rand_result2.numpy()),
        f"\ninput:\n{input}\nrand_result1:\n{rand_result1}\nrand_result2:\n{rand_result2}",
    )


def _get_shape_and_device_from_args(pop_device=False, **kwargs):
    if "size" in kwargs:
        shape = kwargs["size"]
    elif "shape" in kwargs:
        shape = kwargs["shape"]
    elif "n" in kwargs:
        shape = (kwargs["n"],)
    else:
        raise ValueError(f"can't parse shape from kwargs {kwargs}")

    device = "cpu"
    if "device" in kwargs:
        device = kwargs["device"]

    return shape, device


# Test FRB (Forward Recomputation Backpropagation)
# y = rand_op(x) * w
# dw = fake_rand_op(x) * dy
# (y * w).backward() will result in dy == w
# so dw == y demand rand_op(x) == fake_rand_op(x)
# in checkpoint activation graph
# fake_rand_op in backward should produce the same result with rand_op in forward
def _test_rand_op_in_FRB(test_case, rand_op, input=None, **kwargs):
    rand_op, kwargs = _inspect_rand_op_and_args(rand_op, **kwargs)

    class CheckpointActivationModule(nn.Module):
        def __init__(self, weight, is_src_rand=False):
            super().__init__()
            self.rand_op = rand_op
            self.is_src_rand = is_src_rand
            self.weight = weight
            self.param = nn.Parameter(flow.zeros(*weight.shape))

        def forward(self, x):
            weight = self.param - self.weight
            if self.is_src_rand:
                y = self.rand_op(**kwargs) + x
            else:
                y = self.rand_op(x, **kwargs)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return y * weight

    class TestGraph(nn.Graph):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.to(nn.graph.GraphModule).activation_checkpointing = True
            self.add_optimizer(flow.optim.SGD(self.model.parameters(), lr=1.0))

        def build(self, x):
            y = self.model(x)
            (y * self.model.weight).sum().backward()
            return y

    if input is None:
        shape, device = _get_shape_and_device_from_args(**kwargs)
        x = flow.randn(*shape).to(device)
        weight = flow.randn(*shape).to(device)
        model = CheckpointActivationModule(weight, True).to(device)
        graph = TestGraph(model)
    else:
        x = input
        weight = flow.randn(*input.shape).to(input.device)
        model = CheckpointActivationModule(weight, False).to(input.device)
        graph = TestGraph(model)

    y = graph(x)

    test_case.assertTrue(
        np.allclose(y.numpy(), model.param.numpy()),
        f"\nx=\n{x.numpy()}\nweight=\n{weight.numpy()}\ny=\n{y.numpy()}\ndweight=\n{model.param.numpy()}",
    )


def _test_split_rand_op_in_graph(test_case, rand_op, input=None, **kwargs):
    rand_op, kwargs = _inspect_rand_op_and_args(rand_op, **kwargs)

    class TestGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.rand_op = rand_op

        def build(self, x):
            x = x.to_global(sbp=flow.sbp.split(0))
            y = self.rand_op(x, **kwargs)
            return y

    class TestGraphWithoutInput(nn.Graph):
        def __init__(self, placement):
            super().__init__()
            self.rand_op = rand_op
            self.placement = placement

        def build(self):
            y = self.rand_op(placement=self.placement, sbp=flow.sbp.split(0), **kwargs)
            return y

    ranks = np.array(range(flow.env.get_world_size()))
    if input is None:
        device = kwargs.pop("device", None)
        placement = flow.placement(device, ranks)
        graph = TestGraphWithoutInput(placement)
        y_global = graph()
    else:
        x = flow.concat([input, input], dim=0)
        placement = flow.placement(input.device.type, ranks)
        # local to broadcast global
        x_global = x.to_global(placement=placement, sbp=flow.sbp.broadcast(), copy=True)
        graph = TestGraph()
        y_global = graph(x_global)

    if isinstance(y_global, (list, tuple)):
        y_global = y_global[0]

    y_global = y_global.to_global(placement=placement, sbp=flow.sbp.broadcast())
    half = y_global.shape[0] // 2
    first_half = y_global[0:half]
    second_half = y_global[half:]
    test_case.assertFalse(np.allclose(first_half.numpy(), second_half.numpy()))


def _test_broadcast_rand_op_in_graph(test_case, rand_op, input=None, **kwargs):
    rand_op, kwargs = _inspect_rand_op_and_args(rand_op, **kwargs)

    class TestGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.rand_op = rand_op

        def build(self, x):
            y = self.rand_op(x, **kwargs)
            return y

    class TestGraphWithoutInput(nn.Graph):
        def __init__(self, placement):
            super().__init__()
            self.rand_op = rand_op
            self.placement = placement

        def build(self):
            y = self.rand_op(placement=placement, sbp=flow.sbp.broadcast(), **kwargs)
            return y

    ranks = np.array(range(flow.env.get_world_size()))
    if input is None:
        device = kwargs.pop("device", None)
        placement = flow.placement(device, ranks)
        graph = TestGraphWithoutInput(placement)
        y_global = graph()
    else:
        placement = flow.placement(input.device.type, ranks)
        # local to broadcast global
        x = input
        x_global = x.to_global(placement=placement, sbp=flow.sbp.broadcast(), copy=True)
        graph = TestGraph()
        y_global = graph(x_global)

    if isinstance(y_global, (list, tuple)):
        y_local = y_global[0].to_local()
    else:
        y_local = y_global.to_local()

    y_all_ranks = y_local.to_global(placement=placement, sbp=flow.sbp.split(0))
    y_allgather = y_all_ranks.to_global(sbp=flow.sbp.broadcast())
    half = y_allgather.shape[0] // 2
    first_half = y_allgather[0:half]
    second_half = y_allgather[half:]
    test_case.assertTrue(np.allclose(first_half.numpy(), second_half.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestRandOpInGraph(oneflow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op(self):
        for device in ("cpu", "cuda"):
            x = flow.randn(4, 16, device=device)
            _test_rand_op_in_graph(self, nn.Dropout, x, p=0.5)
            _test_rand_op_in_graph(self, flow._C.rrelu, x, training=True)
            _test_rand_op_in_graph(self, nn.init.uniform_, x)
            _test_rand_op_in_graph(self, flow._C.exponential_, x)

            x1 = flow.rand(4, 16, device=device)
            _test_rand_op_in_graph(
                self, flow.multinomial, x1, num_samples=16, replacement=True
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op(self):
        shape = (4, 16)
        for device in ("cpu", "cuda"):
            _test_rand_op_in_graph(self, flow.rand, size=shape, device=device)
            _test_rand_op_in_graph(
                self, flow.normal, mean=0.0, std=1.0, size=shape, device=device
            )
            _test_rand_op_in_graph(
                self, flow.randint, low=0, high=10, size=shape, device=device
            )
            _test_rand_op_in_graph(self, flow.randperm, n=32, device=device)

    def test_bernoulli(self):
        x1 = flow.randn(4, 16)
        _test_rand_op_in_graph(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(4, 16)
        _test_rand_op_in_graph(self, flow.bernoulli, x2)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_mask_like(self):
        x = flow.randn(4, 16, 128, 128).to("cuda")
        _test_rand_op_in_graph(
            self,
            flow._C.fused_scale_tril_softmax_mask_scale,
            x,
            p=0.1,
            diagonal=2,
            tril_scale_value=-1000,
        )


@flow.unittest.skip_unless_1n1d()
class TestRandOpInFRB(oneflow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op(self):
        for device in ("cpu", "cuda"):
            x = flow.randn(4, 16, device=device)
            _test_rand_op_in_FRB(self, nn.Dropout, x, p=0.5)
            _test_rand_op_in_FRB(self, flow._C.rrelu, x, training=True)
            _test_rand_op_in_FRB(self, nn.init.uniform_, x)
            _test_rand_op_in_FRB(self, flow._C.exponential_, x)

            x1 = flow.rand(4, 16, device=device)
            _test_rand_op_in_FRB(
                self, flow.multinomial, x1, num_samples=16, replacement=True
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op(self):
        shape = (4, 16)
        for device in ("cpu", "cuda"):
            _test_rand_op_in_FRB(self, flow.rand, size=shape, device=device)
            _test_rand_op_in_FRB(
                self, flow.normal, mean=0.0, std=1.0, size=shape, device=device
            )
            _test_rand_op_in_FRB(
                self, flow.randint, low=0, high=10, size=shape, device=device
            )
            _test_rand_op_in_FRB(self, flow.randperm, n=32, device=device)

    def test_bernoulli(self):
        x1 = flow.randn(4, 16)
        _test_rand_op_in_FRB(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(4, 16)
        _test_rand_op_in_FRB(self, flow.bernoulli, x2)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_mask_like(self):
        x = flow.randn(4, 16, 128, 128).to("cuda")
        _test_rand_op_in_FRB(
            self,
            flow._C.fused_scale_tril_softmax_mask_scale,
            x,
            p=0.1,
            diagonal=0,
            tril_scale_value=-1000,
        )


@flow.unittest.skip_unless_1n2d()
class TestGlobalRandInGraph(oneflow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op_with_split(self):
        x = flow.randn(2, 4, device="cuda")
        _test_split_rand_op_in_graph(self, nn.Dropout, x, p=0.5)
        _test_split_rand_op_in_graph(self, flow._C.rrelu, x, training=True)
        _test_split_rand_op_in_graph(self, nn.init.uniform_, x)
        _test_split_rand_op_in_graph(self, flow._C.exponential_, x)

        x1 = flow.rand(2, 8, device="cuda")
        _test_split_rand_op_in_graph(
            self, flow.multinomial, x1, num_samples=8, replacement=True
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_usual_rand_op_with_broadcast(self):
        x = flow.randn(2, 4, device="cuda")
        _test_broadcast_rand_op_in_graph(self, nn.Dropout, x, p=0.5)
        _test_broadcast_rand_op_in_graph(self, flow._C.rrelu, x, training=True)
        _test_broadcast_rand_op_in_graph(self, nn.init.uniform_, x)
        _test_broadcast_rand_op_in_graph(self, flow._C.exponential_, x)

        x1 = flow.rand(2, 8, device="cuda")
        _test_broadcast_rand_op_in_graph(
            self, flow.multinomial, x1, num_samples=8, replacement=True
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op_with_split(self):
        shape = (4, 4)
        _test_split_rand_op_in_graph(self, flow.rand, size=shape, device="cuda")
        _test_split_rand_op_in_graph(
            self, flow.normal, mean=0.0, std=1.0, size=shape, device="cuda"
        )
        _test_split_rand_op_in_graph(
            self, flow.randint, low=0, high=10, size=shape, device="cuda"
        )
        _test_split_rand_op_in_graph(self, flow.randperm, n=32, device="cuda")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_source_rand_op_with_broadcast(self):
        shape = (4, 4)
        _test_broadcast_rand_op_in_graph(self, flow.rand, size=shape, device="cuda")
        _test_broadcast_rand_op_in_graph(
            self, flow.normal, mean=0.0, std=1.0, size=shape, device="cuda"
        )
        _test_broadcast_rand_op_in_graph(
            self, flow.randint, low=0, high=10, size=shape, device="cuda"
        )
        _test_broadcast_rand_op_in_graph(self, flow.randperm, n=32, device="cuda")

    def test_bernoulli_with_split(self):
        x1 = flow.randn(2, 8)
        _test_split_rand_op_in_graph(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(2, 8)
        _test_split_rand_op_in_graph(self, flow.bernoulli, x2)

    def test_bernoulli_with_broadcast(self):
        x1 = flow.randn(2, 8)
        _test_broadcast_rand_op_in_graph(self, flow.bernoulli, x1, p=0.5)
        x2 = flow.rand(2, 8)
        _test_broadcast_rand_op_in_graph(self, flow.bernoulli, x2)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_random_mask_like_with_split(self):
        x = flow.randn(2, 16, 64).to("cuda")
        _test_split_rand_op_in_graph(
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
        _test_broadcast_rand_op_in_graph(
            self,
            flow._C.fused_scale_tril_softmax_mask_scale,
            x,
            p=0.2,
            diagonal=1,
            tril_scale_value=-100,
        )


if __name__ == "__main__":
    unittest.main()
