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
import functools


class VariableGetterComposite(object):
    def __init__(self):
        self.getter_stack = []

    def __call__(self, var_gen_fn, *args, **kwargs):
        def make_inner(outter, inner):
            @functools.wraps(inner)
            def inner_fn():
                return outter(inner, *args, **kwargs)

            return inner_fn

        fn = var_gen_fn
        for getter in self.getter_stack:
            fn = make_inner(getter, fn)

        return fn()

    def register(self, fn):
        self.getter_stack.append(fn)
