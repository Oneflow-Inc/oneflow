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
import glob

import oneflow as flow
import oneflow.unittest


class MyModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = flow.nn.Parameter(flow.ones(3, 4))

    def forward(self, input):
        return self.param + input


class MyGraph(flow.nn.Graph):
    def __init__(self, module, optimizer, lr_scheduler):
        super().__init__()
        self.m = module
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self, input):
        out = self.m(input)
        out.mean().backward()
        return out


def _rand_input():
    return flow.Tensor(np.random.rand(3, 4).astype(np.float32))


def _get_graph_lrs_from_log(log_path):
    lines = []
    with open(log_path, "rt") as f:
        for line in f:
            lines.append(line.strip())

    lines = lines[1:]
    lrs = []
    for i, line in enumerate(lines):
        step, lr = line.split(",")
        assert int(step) == i
        lrs.append(float(lr))

    return lrs


class _DebugMode(object):
    def __enter__(self):
        os.environ["ONEFLOW_DEBUG_MODE"] = "True"

    def __exit__(self, type, value, traceback):
        del os.environ["ONEFLOW_DEBUG_MODE"]


def _compare_graph_lr_scheduler_with_eager(test_case, **kwargs):
    lr_scheduler_class = kwargs.pop("lr_scheduler", None)
    base_lr = kwargs.pop("base_lr", None)
    iters = kwargs.pop("iters", None)
    rtol = kwargs.pop("rtol", 1e-05)
    atol = kwargs.pop("atol", 1e-08)

    if "warmup_method" in kwargs:
        warmup_method = kwargs.pop("warmup_method", "linear")
        warmup_iters = kwargs.pop("warmup_iters", 5)
        warmup_factor = kwargs.pop("warmup_factor", 0.1)
        warmup_prefix = kwargs.pop("warmup_prefix", False)
        need_warmup = True
    else:
        need_warmup = False

    assert base_lr is not None and iters is not None

    module = MyModule()
    optimizer = flow.optim.SGD([module.param], lr=base_lr)
    lr_scheduler = (
        lr_scheduler_class(optimizer, **kwargs) if lr_scheduler_class else None
    )

    if need_warmup:
        lr_scheduler = flow.optim.lr_scheduler.WarmupLR(
            lr_scheduler or optimizer,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_iters,
            warmup_method=warmup_method,
            warmup_prefix=warmup_prefix,
        )

    graph = MyGraph(module, optimizer, lr_scheduler)

    with _DebugMode():
        for _ in range(iters + 1):
            ret = graph(_rand_input())
            ret.numpy()  # sync for graph finishing

    pid = os.getpid()
    lr_log_file = glob.glob(f"log/*/{pid}-train_step2lr.csv")[0]
    lrs = _get_graph_lrs_from_log(lr_log_file)
    lrs = lrs[:iters]

    optimizer.zero_grad(set_to_none=True)
    eager_lrs = [lr_scheduler.get_last_lr()[0]]
    for _ in range(iters):
        ret = module(_rand_input())
        ret.numpy()
        optimizer.step()
        lr_scheduler.step()
        eager_lrs.append(lr_scheduler.get_last_lr()[0])

    eager_lrs = eager_lrs[:iters]

    test_case.assertTrue(
        np.allclose(lrs, eager_lrs, rtol=rtol, atol=atol),
        f"\ngraph_lrs: {lrs}\nvs.\neager_lrs: {eager_lrs}",
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphLRSchedulerWithEager(flow.unittest.TestCase):
    def test_constant_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=10,
            lr_scheduler=flow.optim.lr_scheduler.ConstantLR,
            factor=0.1,
            total_iters=10,
        )

    def test_linear_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.LinearLR,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10,
        )

    def test_linear_lr_end_factor(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.LinearLR,
            start_factor=0.1,
            end_factor=0.9,
            total_iters=10,
        )

    def test_step_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=30,
            lr_scheduler=flow.optim.lr_scheduler.StepLR,
            step_size=10,
            gamma=0.1,
        )

    def test_multi_step_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.MultiStepLR,
            milestones=[5, 15],
            gamma=0.2,
        )

    def test_polynomial_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.PolynomialLR,
            decay_batch=20,
            end_learning_rate=1e-5,
            power=2.0,
            atol=1e-5,
        )
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.01,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.PolynomialLR,
            decay_batch=20,
            end_learning_rate=1e-4,
            power=1.0,
            cycle=True,
        )

    def test_exponential_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=10,
            lr_scheduler=flow.optim.lr_scheduler.ExponentialLR,
            gamma=0.5,
            atol=1e-5,
        )

    def test_cosine_decay_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.CosineDecayLR,
            decay_steps=10,
            alpha=1e-3,
            atol=1e-5,
        )

    def test_cosine_annealing_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.CosineAnnealingLR,
            T_max=10,
            eta_min=1e-4,
            atol=1e-5,
        )

    def test_linear_warmup_cosine_annealing_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.CosineAnnealingLR,
            T_max=20,
            eta_min=1e-5,
            warmup_method="linear",
            warmup_factor=0.1,
            warmup_iters=5,
            warmup_prefix=False,
            atol=1e-5,
        )

    def test_linear_warmup_prefix_cosine_annealing_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.CosineAnnealingLR,
            T_max=20,
            eta_min=1e-5,
            warmup_method="linear",
            warmup_factor=0.1,
            warmup_iters=5,
            warmup_prefix=True,
            atol=1e-5,
        )

    def test_linear_warmup_multistep_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.MultiStepLR,
            milestones=[10, 15],
            gamma=0.1,
            warmup_method="linear",
            warmup_factor=0.1,
            warmup_iters=5,
        )

    def test_constant_warmup_cosine_decay_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.CosineDecayLR,
            decay_steps=20,
            alpha=1e-3,
            warmup_method="constant",
            warmup_factor=0.1,
            warmup_iters=5,
            atol=1e-5,
        )

    def test_constant_warmup_prefix_cosine_decay_lr(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=20,
            lr_scheduler=flow.optim.lr_scheduler.CosineDecayLR,
            decay_steps=20,
            alpha=1e-3,
            warmup_method="constant",
            warmup_factor=0.1,
            warmup_iters=5,
            warmup_prefix=True,
            atol=1e-5,
        )

    def test_only_warmup(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=10,
            lr_scheduler=None,
            warmup_method="linear",
            warmup_factor=0.1,
            warmup_iters=5,
        )

    def test_warmup_iters_equal_to_zero(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=10,
            lr_scheduler=flow.optim.lr_scheduler.StepLR,
            step_size=3,
            gamma=0.5,
            warmup_method="linear",
            warmup_iters=0,
        )

    def test_cosine_annealing_warm_restarts(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=50,
            lr_scheduler=flow.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            T_0=10,
            T_mult=1,
            eta_min=0.01,
            atol=1e-5,
        )

    def test_cosine_annealing_warm_restarts_mult_2(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=70,
            lr_scheduler=flow.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            T_0=10,
            T_mult=2,
            eta_min=0.01,
            atol=1e-5,
        )

    def test_cosine_annealing_warm_restarts_limit(self):
        _compare_graph_lr_scheduler_with_eager(
            self,
            base_lr=0.1,
            iters=50,
            lr_scheduler=flow.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            T_0=10,
            T_mult=2,
            eta_min=0.01,
            decay_rate=0.5,
            restart_limit=2,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
