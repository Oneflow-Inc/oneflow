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
import numpy as np
import unittest
import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

# y1 = rand_op1(x)
# y2 = rand_op2(x)
# rand_op1 and rand_op2 should have different seed in graph, lead to different result
def _test_rand_in_graph(test_case, rand_op, *args, **kwargs):
    if issubclass(rand_op, nn.Module):
        rand_op1 = rand_op(*args, **kwargs)
        rand_op2 = rand_op(*args, **kwargs)
    else:
        rand_op1 = rand_op
        rand_op2 = rand_op

    class TestGraph(nn.Graph):
        def build(self, x):
            y1 = rand_op1(x)
            y2 = rand_op2(x)
            return y1, y2

    graph = TestGraph()
    input = flow.rand(2, 8)
    rand_result1, rand_result2 = graph(input)
    test_case.assertFalse(np.allclose(rand_result1.numpy(), rand_result2.numpy()))


# y = rand_op(x) * w
# dw = fake_rand_op(x) * dy
# (y * w).backward() will result in dy == w
# so dw == y demand rand_op(x) == fake_rand_op(x)
# in checkpoint activation graph
# fake_rand_op in backward should produce the same result with rand_op in forward
def _test_rand_in_checkpoint_activation_graph(test_case, rand_op, *args, **kwargs):
    if issubclass(rand_op, nn.Module):
        rand_op = rand_op(*args, **kwargs)

    class CheckpointActivationModule(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
            self.param = nn.Parameter(flow.zeros(*weight.shape))

        def forward(self, x):
            weight = self.param - self.weight
            return rand_op(x) * weight

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

    x = flow.rand(2, 4)
    weight = flow.rand(2, 4)

    model = CheckpointActivationModule(weight)
    graph = TestGraph(model)
    y = graph(x)

    # print(x.numpy())
    # print(weight.numpy())
    # print(y.numpy())
    # print(model.param.numpy())
    test_case.assertTrue(np.allclose(y.numpy(), model.param.numpy()))


def _test_split_rand_in_graph(test_case, device, rand_op, *args, **kwargs):
    if issubclass(rand_op, nn.Module):
        rand_op = rand_op(*args, **kwargs)

    placement = flow.placement(device, np.array(range(flow.env.get_world_size())))

    class TestGraph(nn.Graph):
        def build(self, x):
            y = rand_op(x)
            return y

    x = flow.rand(2, 4)
    x_db = flow.concat([x, x], dim=0)
    x_global = x_db.to_global(placement=placement, sbp=flow.sbp.broadcast())
    x_global = x_global.to_global(placement=placement, sbp=flow.sbp.split(0))
    graph = TestGraph()
    y_global = graph(x_global)
    y_global = y_global.to_global(placement=placement, sbp=flow.sbp.broadcast())

    first_half = y_global[0:2, :]
    second_half = y_global[2:, :]
    test_case.assertFalse(np.allclose(first_half.numpy(), second_half.numpy()))


def _test_broadcast_rand_in_graph(test_case, device, rand_op, *args, **kwargs):
    if issubclass(rand_op, nn.Module):
        rand_op = rand_op(*args, **kwargs)

    placement = flow.placement(device, np.array(range(flow.env.get_world_size())))

    class TestGraph(nn.Graph):
        def build(self, x):
            y = rand_op(x)
            return y

    x = flow.rand(2, 4)
    x_global = x.to_global(placement=placement, sbp=flow.sbp.broadcast())
    graph = TestGraph()
    # broadcast with shape (2, 4)
    y = graph(x_global)
    y_local = y.to_local()
    # split with shape (4, 4)
    y_global = y_local.to_global(placement=placement, sbp=flow.sbp.split(0))
    y_global = y_global.to_global(sbp=flow.sbp.broadcast())

    first_half = y_global[0:2, :]
    second_half = y_global[2:, :]
    test_case.assertTrue(np.allclose(first_half.numpy(), second_half.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestRandInGraph(oneflow.unittest.TestCase):
    def test_rand_in_graph(self):
        _test_rand_in_graph(self, nn.Dropout, p=0.5)

    def test_rand_in_checkpoint_activation_graph(self):
        _test_rand_in_checkpoint_activation_graph(self, nn.Dropout, p=0.5)


@flow.unittest.skip_unless_1n2d()
class TestGlobalRandInGraph(oneflow.unittest.TestCase):
    def test_global_rand_in_graph(self):
        _test_split_rand_in_graph(self, "cuda", nn.Dropout, p=0.5)
        _test_broadcast_rand_in_graph(self, "cuda", nn.Dropout, p=0.5)


if __name__ == "__main__":
    unittest.main()
