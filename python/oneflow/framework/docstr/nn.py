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
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.nn.ModuleList,
    r"""ModuleList(modules=None)

    Holds submodules in a list.

    :attr:`ModuleList` can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all :attr:`Module` methods.

    Args:
        modules (Iterable[oneflow.nn.module.Module], optional) - an iterable of modules to add

    For example:

    .. code-block:: python

        import oneflow as flow
        class ModuleListModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleList([SubModule0() for i in range(3)])

            def forward(self, x):
                for i, _ in enumerate(self.linears):
                    x = self.linears[i](x)
                return x
    
    """
)
