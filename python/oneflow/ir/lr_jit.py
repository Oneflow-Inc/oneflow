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
import ast
import textwrap
import inspect
import oneflow

import unittest
import oneflow.unittest

from ast_gen_transformer import ASTTransformer
from math_params_transformer import MathParamsTransformer
from self_params_transformer import SelfParamsTransformer
from bisect_transformer import BisectTransformer


def lr_jit_register(lr_obj, is_dump=False):
    _id = lr_obj.__class__.__name__
    # load source txt
    _src = textwrap.dedent(inspect.getsource(lr_obj.get_lr))
    _ast = ast.parse(_src).body[0]

    # transform param self
    transformer = SelfParamsTransformer(lr_obj)
    transformer.visit(_ast)

    # transform for bisect lib
    transformer = BisectTransformer()
    transformer.visit(_ast)

    # transform for math lib
    transformer = MathParamsTransformer()
    transformer.visit(_ast)

    # feed transformed as to C++
    transformer = ASTTransformer()
    transformer.visit(_ast)

    oneflow._oneflow_internal.ir.compile_and_register_lr_jit(_id, _ast.ast, is_dump)
    return _id


def _test_current_lr_jit(test_case):
    from oneflow.nn.optimizer.constant_lr import ConstantLR
    from oneflow.nn.optimizer.cosine_annealing_lr import CosineAnnealingLR
    from oneflow.nn.optimizer.cosine_decay_lr import CosineDecayLR
    from oneflow.nn.optimizer.exponential_lr import ExponentialLR
    from oneflow.nn.optimizer.lambda_lr import LambdaLR
    from oneflow.nn.optimizer.linear_lr import LinearLR
    from oneflow.nn.optimizer.multistep_lr import MultiStepLR
    from oneflow.nn.optimizer.polynomial_lr import PolynomialLR
    from oneflow.nn.optimizer.sequential_lr import SequentialLR
    from oneflow.nn.optimizer.step_lr import StepLR
    from oneflow.nn.optimizer.warmup_lr import WarmupLR

    from oneflow.optim import SGD
    from oneflow.nn import Parameter
    import numpy as np

    param = Parameter(oneflow.ones(3, 4))
    optimizer = SGD([param], lr=0.001)

    lr_jit = oneflow._oneflow_internal.ir.create_global_lr_jit()

    lr_obj_list = [
        # WarmupLR(optimizer),
        StepLR(optimizer, 5),
        # SequentialLR(optimizer),
        PolynomialLR(optimizer, 5),
        MultiStepLR(optimizer, [10, 20, 30]),
        LinearLR(optimizer),
        # LambdaLR(optimizer, [lambda step: 0.95 * step]),
        ExponentialLR(optimizer, 1.1),
        CosineDecayLR(optimizer, 10),
        CosineAnnealingLR(optimizer, 50),
        ConstantLR(optimizer),
    ]

    for lr_obj in lr_obj_list:
        id_ = lr_jit_register(lr_obj, False)

        ls = [[0.005, 5], [0.01, 10], [0.02, 21]]
        for elem in ls:
            base_lr = elem[0]
            step = elem[1]
            lr = lr_obj.get_lr(base_lr, step)
            lr_jit = oneflow._oneflow_internal.ir.get_lr(id_, base_lr, step)
            test_case.assertTrue(np.isclose(lr, lr_jit))


@oneflow.unittest.skip_unless_1n1d()
class TestCurrentLRJIT(oneflow.unittest.TestCase):
    def test_current_lr_jit(test_case):
        _test_current_lr_jit(test_case)


if __name__ == "__main__":
    unittest.main()
