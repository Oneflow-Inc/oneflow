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
        if node.value.id == 'self':
            _name = node.attr
            _attr = getattr(self.lr_def_class, _name)
            return ast.Constant(_attr, None)

def lr_def(func):
    @wraps(func)
    def wrapper(self, base_lr:float, step:int, last_lr:float=0):
        _id = self.__class__.__name__
        _src = textwrap.dedent(inspect.getsource(func))
        _ast = ast.parse(_src)
        transformer = SelfParamsTransformer(self)
        print(ast.dump(_ast))
        transformer.visit(_ast)
        print(ast.dump(_ast))
        # oneflow._oneflow_internal.ir.compile_and_register_lr_jit(_ast, _id)

    return wrapper



if __name__ == "__main__":
    class TestJITLR:
        var = 1
        @lr_def
        def get_lr(self, base_lr:float, step:int):
            return base_lr*step*self.var*3

    TestJITLR().get_lr(0.01, 2)
