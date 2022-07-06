import ast


class SelfParamsTransformer(ast.NodeTransformer):
    def __init__(self, lr_obj):
        super().__init__()
        self.lr_obj = lr_obj

    def visit_Attribute(self, node):
        if node.value.id == "self":
            _name = node.attr
            _attr = getattr(self.lr_obj, _name)
            if isinstance(_attr, list):
                ls = [ast.Constant(elem, None) for elem in _attr]
                return ast.List(ls)
            return ast.Constant(_attr, None)
        return node

    def visit_arguments(self, node: ast.arguments):
        for index, item in enumerate(node.args):
            if item.arg == "self":
                node.args.pop(index)
        return node
