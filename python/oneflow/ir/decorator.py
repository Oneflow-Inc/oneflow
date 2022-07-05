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


class ASTTransformer(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg):
        node.ast = oneflow._oneflow_internal.ir.arg_(node.arg)
        return node

    def visit_arguments(self, node: ast.arguments):
        for arg in node.args:
            self.visit(arg)

        list = [ arg.ast for arg in node.args]
        node.ast = oneflow._oneflow_internal.ir.arguments_(list)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in node.body:
            self.visit(arg)

        body = [ arg.ast for arg in node.body]
        self.visit(node.args)
        node.ast = oneflow._oneflow_internal.ir.FunctionDef_("get_lr", node.args.ast, body)
        return node

    def visit_Return(self, node: ast.Return):
        self.visit(node.value)

        node.ast = oneflow._oneflow_internal.ir.Return_(node.value.ast)
        return node

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for arg in node.targets:
            self.visit(arg)

        targets = [ arg.ast for arg in node.targets]
        node.ast = oneflow._oneflow_internal.ir.Assign_(targets, node.value.ast)
        return node



    def visit_If(self, node: ast.If):
        self.visit(node.test)
        for arg in node.body:
            self.visit(arg)

        for arg in node.orelse:
            self.visit(arg)

        test = node.test.ast
        body = [ arg.ast for arg in node.body]
        orelse = [ arg.ast for arg in node.orelse]
        node.ast = oneflow._oneflow_internal.ir.If_(test, body, orelse)
        return node

    def visit_Raise(self, node: ast.Raise):
        print(ast.dump(node))
        raise "not suport yet now"

    def visit_Assert(self, node: ast.Assert):
        print(ast.dump(node))
        raise "not suport yet now"

    def visit_Expr(self, node: ast.Expr):
        print(ast.dump(node))
        raise "not suport yet now"

    def visit_BoolOp(self, node: ast.BoolOp):
        print(ast.dump(node))
        raise "not suport yet now"

    def visit_BinOp(self, node: ast.BinOp):
        self.visit(node.left)
        self.visit(node.right)

        left = node.left.ast
        right = node.right.ast

        def get_op(op: ast.operator):
            list = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow]
            res = 1
            for elem in list:
                if isinstance(op, elem):
                    return res
                res += 1

        op = get_op(node.op)

        node.ast = oneflow._oneflow_internal.ir.BinOp_(left, op, right)
        return node

    def visit_Lambda(self, node: ast.Lambda):
        print(ast.dump(node))
        raise "not suport yet now"

    def visit_Compare(self, node: ast.Compare):
        self.visit(node.left)

        for arg in node.comparators:
            self.visit(arg)

        left = node.left.ast
        comparators = [ arg.ast for arg in node.comparators]

        def get_op(op: ast.operator):
            list = [ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE]
            res = 1
            for elem in list:
                if isinstance(op, elem):
                    return res
                res += 1
        ops = [ get_op(arg) for arg in node.ops ]

        node.ast = oneflow._oneflow_internal.ir.Compare_(left, ops, comparators)
        return node

    def visit_Call(self, node: ast.Call):
        self.visit(node.func)

        for arg in node.args:
            self.visit(arg)

        func = node.func.ast
        args = [ arg.ast for arg in node.args]

        node.ast = oneflow._oneflow_internal.ir.Call_(func, args)
        return node

    def visit_Constant(self, node: ast.Constant):
        node.ast = oneflow._oneflow_internal.ir.Constant_(node.value)
        return node

    def visit_Num(self, node: ast.Num):
        node.ast = oneflow._oneflow_internal.ir.Num_(node.value)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.visit(node.value)
        value = node.value.ast

        node.ast = oneflow._oneflow_internal.ir.Attribute_(value, node.attr)
        return node

    def visit_Name(self, node: ast.Name):
        node.ast = oneflow._oneflow_internal.ir.Name_(node.id)
        return node


    # def visit_Return(self, node: Return) -> Any: ...
    # def visit_Assign(self, node: Assign) -> Any: ...
    # def visit_If(self, node: If) -> Any: ...
    # def visit_Raise(self, node: Raise) -> Any: ...

def lr_jit_register(lr_class):
    _id = lr_class.__class__.__name__
    # load source txt
    _src = textwrap.dedent(inspect.getsource(lr_class.get_lr))
    _ast = ast.parse(_src).body[0]
    # transform param self
    transformer = SelfParamsTransformer(lr_class)
    transformer.visit(_ast)
    transformer = ASTTransformer()
    transformer.visit(_ast)
    # feed transformed as to C++
    print(ast.dump(_ast))
    # _arg = oneflow._oneflow_internal.ir.arg_("test")
    # _args = oneflow._oneflow_internal.ir.arguments_([_arg])
    # _ast = oneflow._oneflow_internal.ir.FunctionDef_("test", _args, [])
    oneflow._oneflow_internal.ir.compile_and_register_lr_jit(_id, _ast.ast)
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

    class Test:
        def get_lr(base_lr:float, step:float):
            return step + base_lr

    id = lr_jit_register(Test)
    res = oneflow._oneflow_internal.ir.get_lr(id, 4, 5)
    print(res)

    # lr_class_list = [
    #     # WarmupLR(optimizer),
    #     # StepLR(optimizer, 5),
    #     # # SequentialLR(optimizer),
    #     # PolynomialLR(optimizer, 5),
    #     # MultiStepLR(optimizer, [10]),
    #     # LinearLR(optimizer),
    #     # LambdaLR(optimizer, [lambda step: 0.95 * step]),
    #     # ExponentialLR(optimizer, 1.1),
    #     # CosineDecayLR(optimizer, 10),
    #     # CosineAnnealingLR(optimizer, 50),
    #     ConstantLR(optimizer)
    # ]
    # for lr_class in lr_class_list:
    #     lr_jit_register(lr_class)
