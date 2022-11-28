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
from oneflow.nn.graph import GraphModule
import oneflow.unittest


class MyModule1(flow.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = flow.nn.Parameter(param)

    def forward(self, input):
        x = flow._C.matmul(input, self.param, transpose_b=True)
        return flow._C.gelu(x)


class MyModule2(flow.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = flow.nn.Parameter(param)

    def forward(self, input, target):
        x = flow._C.matmul(input, self.param)
        loss = flow._C.sparse_softmax_cross_entropy(x, target)
        return loss.mean()
        # return loss


def _make_optimizer(params, norm_type, max_norm):
    return flow.optim.SGD(
        [
            {
                "params": params,
                "lr": 1.0,
                "momentum": 0.0,
                "clip_grad_max_norm": max_norm,
                "clip_grad_norm_type": norm_type,
            },
        ]
    )


class MyGraph(flow.nn.Graph):
    def __init__(self, module1, module2, optimizer=None, acc=1):
        super().__init__()

        self.m1 = module1
        self.m2 = module2

        if (
            module1.param.is_global
            and module2.param.is_global
            and module1.param.placement != module2.param.placement
        ):
            self.m1.to(GraphModule).set_stage(0)
            self.m2.to(GraphModule).set_stage(1)

        if optimizer is not None:
            self.add_optimizer(optimizer)

        if acc > 1:
            self.config.set_gradient_accumulation_steps(acc)

    def build(self, input, target):
        x = self.m1(input)
        if x.is_global and target.is_global and x.placement != target.placement:
            x = x.to_global(placement=target.placement)
        loss = self.m2(x, target)
        loss.backward()
        return loss


class TensorGenerator(object):
    def __init__(
        self, batch_size=8, feat1=10, feat2=8, device="cuda", parallel_mode=None
    ):
        input = flow.randn(batch_size, feat1).to(device)
        param1 = flow.randn(feat2, feat1).to(device)
        param2 = flow.randn(feat2, feat1).to(device)
        target = flow.randint(0, 10, (batch_size,)).to(device)

        ranks = np.array(range(flow.env.get_world_size()))
        placement = flow.placement(device, ranks)
        self.input = input.to_global(placement, sbp=flow.sbp.broadcast)
        self.param1 = param1.to_global(placement, sbp=flow.sbp.broadcast)
        self.param2 = param2.to_global(placement, sbp=flow.sbp.broadcast)
        self.target = target.to_global(placement, sbp=flow.sbp.broadcast)

        self.input_sbp = None
        self.target_sbp = None
        self.param1_sbp = None
        self.param2_sbp = None
        self.placement1 = None
        self.placement2 = None

        if parallel_mode is not None:
            assert isinstance(parallel_mode, str) or isinstance(
                parallel_mode, (list, tuple)
            )

            if isinstance(parallel_mode, str):
                parallel_mode = [parallel_mode]

            assert all(p.upper() in ("DP", "MP", "PP") for p in parallel_mode)
            assert len(parallel_mode) > 0 and len(parallel_mode) <= 2

            self.input_sbp = []
            self.target_sbp = []
            self.param1_sbp = []
            self.param2_sbp = []

            has_pp = False

            for p in parallel_mode:
                if p == "DP":
                    self.input_sbp.append(flow.sbp.split(0))
                    self.target_sbp.append(flow.sbp.split(0))
                    self.param1_sbp.append(flow.sbp.broadcast())
                    self.param2_sbp.append(flow.sbp.broadcast())
                elif p == "MP":
                    self.input_sbp.append(flow.sbp.broadcast())
                    self.target_sbp.append(flow.sbp.broadcast())
                    self.param1_sbp.append(flow.sbp.split(0))
                    self.param2_sbp.append(flow.sbp.split(0))
                elif p == "PP":
                    ranks = ranks.reshape(2, -1)
                    self.placement1 = flow.placement(device, ranks[0])
                    self.placement2 = flow.placement(device, ranks[1])
                    has_pp = True
                else:
                    raise ValueError

            if len(parallel_mode) > 1 and not has_pp:
                ranks = ranks.reshape(2, -1)
                self.placement1 = flow.placement(device, ranks)
                self.placement2 = flow.placement(device, ranks)

            if len(self.input_sbp) == 0:
                self.input_sbp = None

            if len(self.target_sbp) == 0:
                self.target_sbp = None

            if len(self.param1_sbp) == 0:
                self.param1_sbp = None

            if len(self.param2_sbp) == 0:
                self.param2_sbp = None

    def local_input(self):
        return self.input.to_local()

    def local_target(self):
        return self.target.to_local()

    def local_param1(self):
        return self.param1.clone().to_local()

    def local_param2(self):
        return self.param2.clone().to_local()

    def global_input(self):
        if self.input_sbp is None and self.placement1 is None:
            return self.input

        return self.input.to_global(placement=self.placement1, sbp=self.input_sbp)

    def global_target(self):
        if self.target_sbp is None and self.placement2 is None:
            return self.target

        return self.target.to_global(placement=self.placement2, sbp=self.target_sbp)

    def global_param1(self):
        if self.param1_sbp is None and self.placement1 is None:
            return self.param1.clone()

        return self.param1.to_global(placement=self.placement1, sbp=self.param1_sbp)

    def global_param2(self):
        if self.param2_sbp is None and self.placement2 is None:
            return self.param2.clone()

        return self.param2.to_global(placement=self.placement2, sbp=self.param2_sbp)


def _compare_with_eager(
    test_case,
    *,
    batch_size=8,
    acc=1,
    norm_type=2.0,
    max_norm=1.0,
    device="cuda",
    parallel_mode=None,
    rtol=1e-03,
    atol=1e-05,
):
    gen = TensorGenerator(
        batch_size=batch_size, device=device, parallel_mode=parallel_mode
    )

    # eager
    m1 = MyModule1(gen.local_param1())
    m2 = MyModule2(gen.local_param2())
    opt = _make_optimizer([m1.param, m2.param], norm_type, max_norm)
    x = m1(gen.local_input())
    loss = m2(x, gen.local_target())
    opt.zero_grad()
    loss.backward()
    opt.clip_grad()
    opt.step()

    loss_a = loss.numpy()
    grad1_a = m1.param.numpy()
    grad2_a = m2.param.numpy()

    # graph
    graph_m1 = MyModule1(gen.global_param1())
    graph_m2 = MyModule2(gen.global_param2())
    opt = _make_optimizer([graph_m1.param, graph_m2.param], norm_type, max_norm)
    graph = MyGraph(graph_m1, graph_m2, opt, acc)
    graph_loss = graph(gen.global_input(), gen.global_target())

    # debug
    # rank = flow.env.get_rank()
    # print("")
    # print(f"[rank{rank}] eager local loss: {loss}")

    # print(
    #     f"[rank{rank}] graph_loss placement: {graph_loss.placement}, sbp: {graph_loss.sbp}"
    # )
    # print(f"[rank{rank}] graph_loss: {graph_loss}")

    # local_loss = graph_loss.to_local()
    # print(f"[rank{rank}] local_loss.numel(): {local_loss.numel()}")
    # print(f"[rank{rank}] local_loss: {local_loss}")

    if acc > 1 and graph_loss.numel() == acc:
        graph_loss = graph_loss.mean()

    if parallel_mode is None:
        loss_b = graph_loss.numpy()
        grad1_b = graph.m1.to(flow.nn.Module).param.numpy()
        grad2_b = graph.m2.to(flow.nn.Module).param.numpy()
    else:
        ranks = np.array(range(flow.env.get_world_size()))
        placement = flow.placement(device, ranks)
        loss_b = graph_loss.to_global(placement, flow.sbp.broadcast).to_local().numpy()
        grad1_b = graph.m1.to(flow.nn.Module).param.to_global(
            placement, flow.sbp.broadcast
        )
        grad1_b = grad1_b.to_local().numpy()
        grad2_b = graph.m2.to(flow.nn.Module).param.to_global(
            placement, flow.sbp.broadcast
        )
        grad2_b = grad2_b.to_local().numpy()

    # compare
    test_case.assertTrue(
        np.allclose(loss_a, loss_b, rtol=rtol, atol=atol), f"{loss_a} vs. {loss_b}"
    )
    test_case.assertTrue(
        np.allclose(grad1_a, grad1_b, rtol=rtol, atol=atol),
        f"\n{grad1_a}\nvs.\n{grad1_b}",
    )
    test_case.assertTrue(
        np.allclose(grad2_a, grad2_b, rtol=rtol, atol=atol),
        f"\n{grad2_a}\nvs.\n{grad2_b}",
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphClipGradNorm(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_local(test_case):
        _compare_with_eager(test_case)

    @flow.unittest.skip_unless_1n1d()
    def test_acc(test_case):
        _compare_with_eager(test_case, batch_size=8, acc=8)

    @flow.unittest.skip_unless_1n2d()
    def test_dp(test_case):
        _compare_with_eager(test_case, parallel_mode="DP")

    @flow.unittest.skip_unless_1n2d()
    def test_mp(test_case):
        _compare_with_eager(test_case, parallel_mode="MP")

    @flow.unittest.skip_unless_1n2d()
    def test_pp(test_case):
        _compare_with_eager(test_case, parallel_mode="PP")

    @flow.unittest.skip_unless_1n2d()
    def test_pp_acc(test_case):
        _compare_with_eager(test_case, batch_size=8, acc=8, parallel_mode="PP")

    @flow.unittest.skip_unless_1n4d()
    def test_dp_mp(test_case):
        _compare_with_eager(test_case, parallel_mode=["DP", "MP"])

    @flow.unittest.skip_unless_1n4d()
    def test_mp_pp(test_case):
        _compare_with_eager(test_case, parallel_mode=["MP", "PP"])

    @flow.unittest.skip_unless_1n4d()
    def test_dp_pp(test_case):
        _compare_with_eager(test_case, parallel_mode=["DP", "PP"])

    @flow.unittest.skip_unless_1n4d()
    def test_mp_pp_acc(test_case):
        _compare_with_eager(test_case, batch_size=8, acc=8, parallel_mode=["MP", "PP"])

    @flow.unittest.skip_unless_1n4d()
    def test_dp_pp_acc(test_case):
        _compare_with_eager(test_case, batch_size=8, acc=4, parallel_mode=["DP", "PP"])


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphClipGradNormInf(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_local(test_case):
        _compare_with_eager(test_case, norm_type=float("inf"))

    @flow.unittest.skip_unless_1n1d()
    def test_acc(test_case):
        _compare_with_eager(
            test_case, batch_size=8, acc=8, norm_type=-float("inf"), atol=1e-6
        )

    @flow.unittest.skip_unless_1n2d()
    def test_dp(test_case):
        _compare_with_eager(
            test_case,
            norm_type=float("inf"),
            max_norm=2.0,
            parallel_mode="DP",
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n2d()
    def test_mp(test_case):
        _compare_with_eager(
            test_case,
            norm_type=-float("inf"),
            max_norm=3.0,
            parallel_mode="MP",
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n2d()
    def test_pp(test_case):
        _compare_with_eager(
            test_case,
            norm_type=float("inf"),
            max_norm=4.0,
            parallel_mode="PP",
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n2d()
    def test_pp_acc(test_case):
        _compare_with_eager(
            test_case,
            batch_size=8,
            acc=8,
            norm_type=-float("inf"),
            max_norm=5.0,
            parallel_mode="PP",
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n4d()
    def test_dp_mp(test_case):
        _compare_with_eager(
            test_case,
            norm_type=float("inf"),
            max_norm=1.1,
            parallel_mode=["DP", "MP"],
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n4d()
    def test_mp_pp(test_case):
        _compare_with_eager(
            test_case,
            norm_type=-float("inf"),
            max_norm=1.2,
            parallel_mode=["MP", "PP"],
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n4d()
    def test_dp_pp(test_case):
        _compare_with_eager(
            test_case,
            norm_type=float("inf"),
            max_norm=1.3,
            parallel_mode=["DP", "PP"],
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n4d()
    def test_mp_pp_acc(test_case):
        _compare_with_eager(
            test_case,
            batch_size=8,
            acc=8,
            norm_type=float("inf"),
            max_norm=2.1,
            parallel_mode=["MP", "PP"],
            atol=1e-6,
        )

    @flow.unittest.skip_unless_1n4d()
    def test_dp_pp_acc(test_case):
        _compare_with_eager(
            test_case,
            batch_size=8,
            acc=4,
            norm_type=-float("inf"),
            max_norm=2.2,
            parallel_mode=["DP", "PP"],
            atol=1e-6,
        )


if __name__ == "__main__":
    # flow.manual_seed(0)
    unittest.main()
