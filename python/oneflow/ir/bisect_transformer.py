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
from bisect import bisect


class BisectTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.body_index = 0
        self.body = node.body
        for stmt in node.body:
            self.visit(stmt)
        self.body_index += 1
        return node

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            func: ast.Attribute = node.func
            if func.value.id == "bisect":
                bisect_x_list = ["bisect_right", "bisect_left"]
                if func.attr in bisect_x_list:
                    op = ast.LtE
                    if func.attr == "bisect_right":
                        op = ast.Lt
                    if not isinstance(node.args[0], ast.List):
                        raise "only support bisect.bisect_right(list, x)"
                    ls = node.args[0].elts
                    cmp = node.args[1]
                    index = 0
                    for i in ls[::-1]:
                        test = ast.Compare(cmp, [op()], [i])
                        assign = ast.Assign(
                            [ast.Name("tmp")], ast.Constant(len(ls) - index - 1, None)
                        )
                        if "orelse" in locals():
                            orelse = ast.If(test, [assign], [orelse])
                        else:
                            orelse = ast.If(test, [assign], [])
                        index += 1
                    self.body.insert(self.body_index, orelse)
                    return ast.Name("tmp")
        return node
