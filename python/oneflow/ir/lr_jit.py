from functools import wraps
import ast
import textwrap
import inspect
import oneflow

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
    # print(ast.dump(_ast))
    transformer = SelfParamsTransformer(lr_obj)
    transformer.visit(_ast)
    # print(ast.dump(_ast))

    transformer = BisectTransformer()
    transformer.visit(_ast)
    # print(ast.dump(_ast))

    transformer = MathParamsTransformer()
    transformer.visit(_ast)
    # print(ast.dump(_ast))

    transformer = ASTTransformer()
    transformer.visit(_ast)
    # print(ast.dump(_ast))

    # feed transformed as to C++
    oneflow._oneflow_internal.ir.compile_and_register_lr_jit(_id, _ast.ast, is_dump)
    return _id


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

if __name__ == "__main__":
    param = Parameter(oneflow.ones(3, 4))
    optimizer = SGD([param], lr=0.001)

    lr_jit =  oneflow._oneflow_internal.ir.create_global_lr_jit()

    lr_obj_list = [
        # WarmupLR(optimizer),
        StepLR(optimizer, 5),
        # SequentialLR(optimizer),
        PolynomialLR(optimizer, 5),
        MultiStepLR(optimizer, [10, 20, 30]), # biselect
        LinearLR(optimizer),
        # LambdaLR(optimizer, [lambda step: 0.95 * step]),
        ExponentialLR(optimizer, 1.1),
        CosineDecayLR(optimizer, 10),
        CosineAnnealingLR(optimizer, 50),
        ConstantLR(optimizer)
    ]

    for lr_obj in lr_obj_list:
        print(lr_obj.__class__.__name__)
        id_ = lr_jit_register(lr_obj, False)

        ls = [[0.005, 5], [0.01, 10], [0.02, 21]]
        for elem in ls:
            base_lr = elem[0]
            step = elem[1]
            lr = lr_obj.get_lr(base_lr, step)
            lr_jit =  oneflow._oneflow_internal.ir.get_lr(id_, base_lr, step)

            print("lr: ", lr)
            print("lr_jit: ", lr_jit)
