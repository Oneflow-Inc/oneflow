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
import math
import unittest
import os
import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.nn.parameter import Parameter


def _test_linear_graph_train_with_lr_sch(
    test_case, iter_num, device, get_opt_and_lr_sch
):
    def train_with_module(iter_num=3):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, -0.68758)
        flow.nn.init.constant_(linear.bias, 0.23)

        opt, lr_sch = get_opt_and_lr_sch(linear.parameters())

        x = flow.tensor(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )

        def one_iter():
            of_out = linear(x)
            of_out = of_out.sum()

            of_out.backward()
            opt.step()
            if lr_sch is not None:
                lr_sch.step()
            opt.zero_grad()

            return of_out.numpy(), linear.weight.numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    def train_with_graph(iter_num=3):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, -0.68758)
        flow.nn.init.constant_(linear.bias, 0.23)

        opt, lr_sch = get_opt_and_lr_sch(linear.parameters())

        x = flow.tensor(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                if lr_sch is None:
                    self.add_optimizer(opt)
                else:
                    self.add_optimizer(opt, lr_sch=lr_sch)

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        linear_t_g = LinearTrainGraph()

        def one_iter():
            of_graph_out = linear_t_g(x)
            return (
                of_graph_out.numpy(),
                linear_t_g.linear.weight.to(flow.Tensor).numpy(),
            )

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    module_check_list = train_with_module(iter_num)
    graph_check_list = train_with_graph(iter_num)
    for i in range(iter_num):
        # check equal on loss
        test_case.assertTrue(
            np.allclose(
                module_check_list[i][0],
                graph_check_list[i][0],
                rtol=0.00001,
                atol=0.00001,
            )
        )
        # check equal on weight
        test_case.assertTrue(
            np.allclose(
                module_check_list[i][1],
                graph_check_list[i][1],
                rtol=0.00001,
                atol=0.00001,
            )
        )


def _sgd_cosine_fn(parameters):
    of_sgd = flow.optim.SGD(parameters, lr=0.001)
    alpha = 0.5
    decay_steps = 10
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        of_sgd, decay_steps=decay_steps, alpha=alpha
    )
    return of_sgd, cosine_decay_lr


def _sgd_cosine_constant_fn(parameters):
    of_sgd = flow.optim.SGD(parameters, lr=0.001)
    alpha = 0.5
    decay_steps = 10
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        of_sgd, decay_steps=decay_steps, alpha=alpha
    )
    constant_warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_decay_lr, warmup_factor=0.5, warmup_iters=5, warmup_method="constant"
    )
    return of_sgd, constant_warmup_cosine_lr


def _sgd_constant_fn(parameters):
    of_sgd = flow.optim.SGD(parameters, lr=0.001)
    alpha = 0.5
    steps = 10
    constant_warmup_lr = flow.optim.lr_scheduler.WarmUpLR(
        of_sgd, warmup_factor=0.5, warmup_iters=5, warmup_method="constant"
    )
    return of_sgd, constant_warmup_lr


def _sgd_cosine_linear_fn(parameters):
    of_sgd = flow.optim.SGD(parameters, lr=0.001)
    alpha = 0.5
    decay_steps = 10
    cosine_decay_lr = flow.optim.lr_scheduler.CosineDecayLR(
        of_sgd, decay_steps=decay_steps, alpha=alpha
    )
    linear_warmup_cosine_lr = flow.optim.lr_scheduler.WarmUpLR(
        cosine_decay_lr, warmup_factor=0.5, warmup_iters=5, warmup_method="linear"
    )
    return of_sgd, linear_warmup_cosine_lr


def _sgd_linear_fn(parameters):
    of_sgd = flow.optim.SGD(parameters, lr=0.001)
    alpha = 0.5
    steps = 10
    linear_warmup_lr = flow.optim.lr_scheduler.WarmUpLR(
        of_sgd, warmup_factor=0.5, warmup_iters=5, warmup_method="linear"
    )
    return of_sgd, linear_warmup_lr


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearGraphTrainWithCosineLrScheduler(flow.unittest.TestCase):
    def test_graph_cosine(test_case):
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cuda"), _sgd_cosine_fn
        )
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cpu"), _sgd_cosine_fn
        )

    def test_graph_cosine_constant(test_case):
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cuda"), _sgd_cosine_constant_fn
        )
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cpu"), _sgd_cosine_constant_fn
        )

    def test_graph_constant(test_case):
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cuda"), _sgd_constant_fn
        )
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cpu"), _sgd_constant_fn
        )

    def test_graph_cosine_linear(test_case):
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cuda"), _sgd_cosine_linear_fn
        )
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cpu"), _sgd_cosine_linear_fn
        )

    def test_graph_linear(test_case):
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cuda"), _sgd_linear_fn
        )
        _test_linear_graph_train_with_lr_sch(
            test_case, 21, flow.device("cpu"), _sgd_linear_fn
        )


if __name__ == "__main__":
    unittest.main()
