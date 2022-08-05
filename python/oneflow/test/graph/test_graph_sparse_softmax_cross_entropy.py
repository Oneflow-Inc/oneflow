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
import numpy as np
import oneflow as flow
import oneflow.unittest


class CrossEntropyModule(flow.nn.Module):
    def __init__(self, pred):
        super().__init__()
        if pred.is_global:
            self.param = flow.nn.Parameter(
                flow.zeros(
                    *pred.shape,
                    dtype=pred.dtype,
                    placement=pred.placement,
                    sbp=pred.sbp,
                )
            )
        else:
            self.param = flow.nn.Parameter(
                flow.zeros(*pred.shape, dtype=pred.dtype, device=pred.device)
            )

    def forward(self, pred, label):
        pred = pred + self.param
        loss = flow._C.sparse_softmax_cross_entropy(pred, label)
        return loss.mean()


class CrossEntropyGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module
        self.add_optimizer(flow.optim.SGD([module.param], lr=1.0, momentum=0.0))

    def build(self, pred, label):
        loss = self.m(pred, label)
        loss.backward()
        return loss


def _compare_with_nn_cross_entropy_loss(
    test_case, pred, label, pred_sbp=None, label_sbp=None
):
    if pred.is_global:
        assert label.is_global
        pred_ = pred.to_local().detach().clone()
        label_ = label.to_local()
    else:
        pred_ = pred.detach().clone()
        label_ = label

    pred_.requires_grad = True
    cross_entropy_loss = flow.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(pred_, label_)
    loss.backward()

    if pred_sbp is not None:
        pred = pred.to_global(sbp=pred_sbp)

    if label_sbp is not None:
        label = label.to_global(sbp=label_sbp)

    cross_entropy_module = CrossEntropyModule(pred)
    cross_entropy_graph = CrossEntropyGraph(cross_entropy_module)
    graph_loss = cross_entropy_graph(pred, label)

    loss_a = loss.numpy()
    grad_a = pred_.grad.numpy()
    if graph_loss.is_local:
        loss_b = graph_loss.numpy()
        grad_b = -cross_entropy_module.param.numpy()
    else:
        graph_loss = graph_loss.to_global(
            sbp=[flow.sbp.broadcast()] * len(graph_loss.sbp)
        )
        loss_b = graph_loss.to_local().numpy()
        pred_grad = cross_entropy_module.param.to_global(
            sbp=[flow.sbp.broadcast()] * len(cross_entropy_module.param.sbp)
        )
        grad_b = -pred_grad.to_local().numpy()

    test_case.assertTrue(np.allclose(loss_a, loss_b), f"{loss_a} vs. {loss_b}")
    test_case.assertTrue(np.allclose(grad_a, grad_b), f"\n{grad_a}\nvs.\n{grad_b}")


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestSparseSoftmaxCrossEntropyGraph(oneflow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_local(test_case):
        pred = flow.randn(8, 10).to("cuda")
        label = flow.randint(0, 10, (8,)).to("cuda")
        _compare_with_nn_cross_entropy_loss(test_case, pred, label)

    @flow.unittest.skip_unless_1n2d()
    def test_data_split(test_case):
        pred = flow.randn(8, 10)
        label = flow.randint(0, 10, (8,))
        placement = flow.placement("cuda", list(range(flow.env.get_world_size())))
        pred = pred.to_global(placement=placement, sbp=flow.sbp.broadcast())
        label = label.to_global(placement=placement, sbp=flow.sbp.broadcast())
        _compare_with_nn_cross_entropy_loss(
            test_case, pred, label, flow.sbp.split(0), flow.sbp.split(0)
        )

    @flow.unittest.skip_unless_1n2d()
    def test_model_split(test_case):
        pred = flow.randn(8, 10)
        label = flow.randint(0, 10, (8,))
        placement = flow.placement("cuda", list(range(flow.env.get_world_size())))
        pred = pred.to_global(placement=placement, sbp=flow.sbp.broadcast())
        label = label.to_global(placement=placement, sbp=flow.sbp.broadcast())
        _compare_with_nn_cross_entropy_loss(
            test_case, pred, label, flow.sbp.split(1), flow.sbp.broadcast()
        )

    @flow.unittest.skip_unless_1n4d()
    def test_2d_split(test_case):
        pred = flow.randn(8, 10)
        label = flow.randint(0, 10, (8,))
        placement = flow.placement(
            "cuda", np.array(range(flow.env.get_world_size())).reshape(2, 2)
        )
        pred = pred.to_global(
            placement=placement, sbp=[flow.sbp.broadcast(), flow.sbp.broadcast()]
        )
        label = label.to_global(
            placement=placement, sbp=[flow.sbp.broadcast(), flow.sbp.broadcast()]
        )
        _compare_with_nn_cross_entropy_loss(
            test_case,
            pred,
            label,
            [flow.sbp.split(0), flow.sbp.split(1)],
            [flow.sbp.split(0), flow.sbp.broadcast()],
        )


if __name__ == "__main__":
    unittest.main()
