from functools import wraps
import ast
import textwrap
import inspect
import oneflow


class SelfParamsTransformer(ast.NodeTransformer):
    def __init__(self, lr_def_class):
        super().__init__()
        self.lr_def_class = lr_def_class

    def visit_Attribute(self, node):
        if node.value.id == "self":
            _name = node.attr
            _attr = getattr(self.lr_def_class, _name)
            return ast.Constant(_attr, None)

    def visit_arguments(self, node: ast.arguments):
        for index, item in enumerate(node.args):
            if item.arg == "self":
                node.args.pop(index)
        return node


def lr_jit_register(lr_class):
    _id = lr_class.__class__.__name__
    # load source txt
    _src = textwrap.dedent(inspect.getsource(lr_class.get_lr))
    _ast = ast.parse(_src).body[0]
    # transform param self
    transformer = SelfParamsTransformer(lr_class)
    transformer.visit(_ast)
    # feed transformed as to C++
    res = oneflow._oneflow_internal.ir.compile_and_register_lr_jit(_ast, _id)


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

    lr_class_list = [
        WarmupLR(optimizer),
        # StepLR(optimizer, 5),
        # SequentialLR(optimizer),
        # PolynomialLR(optimizer, 5),
        # MultiStepLR(optimizer, [10]),
        # LinearLR(optimizer),
        # LambdaLR(optimizer, [lambda step: 0.95 * step]),
        # ExponentialLR(optimizer, 1.1),
        # CosineDecayLR(optimizer, 10),
        # CosineAnnealingLR(optimizer, 50),
        # ConstantLR(optimizer)
    ]
    for lr_class in lr_class_list:
        lr_jit_register(lr_class)
