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
from collections import defaultdict
from typing import List, Union

import torch
from oneflow.framework.infer_compiler.with_oneflow_compile import \
    DeployableModule

_nested_counter = defaultdict(lambda: 0)


class TensorInplaceAssign:
    r"""
    This class is used as a context manager, instantiated with either a `torch.nn.Module` or
    `onediff.infer_compiler.with_oneflow_compile.DeployableModule` during initialization.
    Within the context manager, all Tensors associated with the provided module will be
    transformed into AutoInplaceCopyTensor. After transformed, assignments to Tensor.data are
    modified to in-place copying.

    The class is commonly used to ensure the stability of the data_ptr() for weights,
    particularly in scenarios involving the loading of LoRA weights.

    For example:
        >>> class EagerModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.linear1 = torch.nn.Linear(3, 3)
        >>>         self.linear2 = torch.nn.Linear(3, 3)
        >>>
        >>>     def forward(self, x):
        >>>         return self.linear2(self.linear1(x))
        >>>
        >>> eager = EagerModule()
        >>> dptr1 = eager.linear1.weight.data.data_ptr()
        >>> dptr2 = eager.linear2.weight.data.data_ptr()
        >>>
        >>> with TensorInplaceAssign(eager.linear1):
        >>>     eager.linear1.weight.data = torch.randn(3, 3)
        >>>     eager.linear2.weight.data = torch.randn(3, 3)
        >>>
        >>> eager.linear1.weight.data.data_ptr() == dptr1, eager.linear2.weight.data.data_ptr() == dptr2
        (True, False)
    """

    def __init__(
        self, *modules: List[Union[torch.nn.Module, DeployableModule]]
    ) -> None:
        self.modules = set()
        for module in modules:
            if isinstance(module, torch.nn.Module):
                self.modules.add(module)
            elif isinstance(module, DeployableModule):
                self.modules.add(module._deployable_module_model._torch_module)
            else:
                raise TypeError(
                    "TensorInplaceAssign can only accept torch.nn.Module or DeployableModule"
                )

    def __enter__(self):
        global _nested_counter
        for module in self.modules:
            if _nested_counter[module] == 0:
                module.apply(module_convert_parameter)
            _nested_counter[module] += 1

    def __exit__(self, exc_type, exc_value, traceback):
        global _nested_counter
        for module in list(self.modules):
            _nested_counter[module] -= 1
            if _nested_counter[module] == 0:
                module.apply(module_unconvert_parameter)
                _nested_counter.pop(module)
                self.modules.remove(module)


class AutoInplaceCopyTensor(torch.Tensor):
    @property
    def data(self):
        return AutoInplaceCopyTensor(self)

    @data.setter
    def data(self, new_tensor):
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError(
                f"Cannot assign type {type(new_tensor)} to AutoInplaceCopyTensor"
            )
        self.copy_(new_tensor.detach())


class AutoInplaceCopyParameter(torch.nn.Parameter):
    @property
    def data(self):
        return AutoInplaceCopyTensor(super(AutoInplaceCopyParameter, self).data)

    @data.setter
    def data(self, new_tensor):
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError(
                f"Cannot assign type {type(new_tensor)} to AutoInplaceCopyParameter"
            )
        self.data.copy_(new_tensor.detach())


def module_convert_parameter(module: torch.nn.Module) -> torch.nn.Module:
    for k, v in module.__dict__.items():
        if isinstance(v, torch.nn.Parameter):
            module.__dict__[k] = AutoInplaceCopyParameter(v)
        elif isinstance(v, torch.Tensor):
            module.__dict__[k] = AutoInplaceCopyTensor(v)
    for k, param in module._parameters.items():
        if not isinstance(param, AutoInplaceCopyParameter) and param is not None:
            module._parameters[k] = AutoInplaceCopyParameter(param)
    for k, buffer in module._buffers.items():
        if not isinstance(buffer, AutoInplaceCopyTensor) and buffer is not None:
            module._buffers[k] = AutoInplaceCopyTensor(buffer)
    return module


def module_unconvert_parameter(module: torch.nn.Module) -> torch.nn.Module:
    for k, v in module.__dict__.items():
        if isinstance(v, AutoInplaceCopyParameter):
            module.__dict__[k] = torch.nn.Parameter(torch.Tensor(v.data))
        elif isinstance(v, AutoInplaceCopyTensor):
            module.__dict__[k] = torch.Tensor(v)
    for k, param in module._parameters.items():
        if isinstance(param, AutoInplaceCopyParameter):
            module._parameters[k] = torch.nn.Parameter(torch.Tensor(param.data))
    for k, buffer in module._buffers.items():
        if isinstance(buffer, AutoInplaceCopyTensor):
            module._buffers[k] = torch.Tensor(buffer)
    return module


if __name__ == "__main__":

    class EagerModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(3, 3)
            self.linear2 = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    eager = EagerModule()
    dptr1 = eager.linear1.weight.data.data_ptr()
    dptr2 = eager.linear2.weight.data.data_ptr()

    with TensorInplaceAssign(eager):
        eager.linear1.weight.data = torch.randn(3, 3)
        eager.linear2.weight.data = torch.randn(3, 3)

    assert dptr1 == eager.linear1.weight.data.data_ptr()
    assert dptr2 == eager.linear2.weight.data.data_ptr()

    dptr1 = eager.linear1.weight.data.data_ptr()
    dptr2 = eager.linear2.weight.data.data_ptr()
    with TensorInplaceAssign(eager.linear1):
        eager.linear1.weight.data = torch.randn(3, 3)
        eager.linear2.weight.data = torch.randn(3, 3)
    assert dptr1 == eager.linear1.weight.data.data_ptr()
    assert dptr2 != eager.linear2.weight.data.data_ptr()

    dptr1 = eager.linear1.weight.data.data_ptr()
    dptr2 = eager.linear2.weight.data.data_ptr()
    with TensorInplaceAssign(eager.linear1):
        with TensorInplaceAssign(eager.linear2):
            with TensorInplaceAssign(eager.linear1):
                pass
            eager.linear1.weight.data = torch.randn(3, 3)
            eager.linear2.weight.data = torch.randn(3, 3)
    assert dptr1 == eager.linear1.weight.data.data_ptr()
    assert dptr2 == eager.linear2.weight.data.data_ptr()
