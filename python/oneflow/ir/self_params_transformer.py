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
