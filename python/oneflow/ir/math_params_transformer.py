import ast

class MathParamsTransformer(ast.NodeTransformer):
    def visit_Attribute(self, node):
        import math
        list = ["pi"]
        if node.value.id == "math":
            if node.attr in list:
                _name = node.attr
                _attr = getattr(math, _name)
                return ast.Constant(_attr, None)
        return node
