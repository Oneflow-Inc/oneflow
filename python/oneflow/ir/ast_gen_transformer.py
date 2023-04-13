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
import oneflow
import ast


class ASTTransformer(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg):
        node.ast = oneflow._oneflow_internal.ir.arg_(node.arg)
        return node

    def visit_arguments(self, node: ast.arguments):
        for arg in node.args:
            self.visit(arg)

        list = [arg.ast for arg in node.args]
        node.ast = oneflow._oneflow_internal.ir.arguments_(list)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in node.body:
            self.visit(arg)

        body = [arg.ast for arg in node.body]
        self.visit(node.args)
        node.ast = oneflow._oneflow_internal.ir.FunctionDef_(
            "get_lr", node.args.ast, body
        )
        return node

    def visit_Return(self, node: ast.Return):
        self.visit(node.value)

        node.ast = oneflow._oneflow_internal.ir.Return_(node.value.ast)
        return node

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for arg in node.targets:
            self.visit(arg)

        targets = [arg.ast for arg in node.targets]
        node.ast = oneflow._oneflow_internal.ir.Assign_(targets, node.value.ast)
        return node

    def visit_If(self, node: ast.If):
        self.visit(node.test)
        for arg in node.body:
            self.visit(arg)

        if node.orelse:
            for arg in node.orelse:
                self.visit(arg)

        test = node.test.ast
        body = [arg.ast for arg in node.body]
        orelse = [arg.ast for arg in node.orelse]
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
        comparators = [arg.ast for arg in node.comparators]

        def get_op(op: ast.operator):
            list = [ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE]
            res = 1
            for elem in list:
                if isinstance(op, elem):
                    return res
                res += 1

        ops = [get_op(arg) for arg in node.ops]

        node.ast = oneflow._oneflow_internal.ir.Compare_(left, ops, comparators)
        return node

    def visit_Call(self, node: ast.Call):
        self.visit(node.func)

        for arg in node.args:
            self.visit(arg)

        func = node.func.ast
        args = [arg.ast for arg in node.args]

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
