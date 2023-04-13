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
import collections.abc
import warnings
import operator
from collections import OrderedDict, abc as container_abcs
from itertools import islice
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
    Generic,
)

import oneflow as flow
from oneflow.nn.modules.module import Module

T = TypeVar("T")


def get_seq(T):
    class SequentialContainer(T):
        @overload
        def __init__(self, *args: T) -> None:
            ...

        @overload
        def __init__(self, arg: "OrderedDict[str, T]") -> None:
            ...

        def __init__(self, *args: Any):
            super(SequentialContainer, self).__init__()
            if len(args) == 1 and isinstance(args[0], OrderedDict):
                for (key, module) in args[0].items():
                    self.add_module(key, module)
            else:
                for (idx, module) in enumerate(args):
                    self.add_module(str(idx), module)

        def _get_item_by_idx(self, iterator, idx):
            """Get the idx-th item of the iterator"""
            size = len(self)
            idx = operator.index(idx)
            if not -size <= idx < size:
                raise IndexError("index {} is out of range".format(idx))
            idx %= size
            return next(islice(iterator, idx, None))

        def __getitem__(self: T, idx) -> T:
            if isinstance(idx, slice):
                return self.__class__(OrderedDict(list(self._modules.items())[idx]))
            else:
                return self._get_item_by_idx(self._modules.values(), idx)

        def __setitem__(self, idx: int, module: T) -> None:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            return setattr(self, key, module)

        def __delitem__(self, idx: Union[slice, int]) -> None:
            if isinstance(idx, slice):
                for key in list(self._modules.keys())[idx]:
                    delattr(self, key)
            else:
                key = self._get_item_by_idx(self._modules.keys(), idx)
                delattr(self, key)

        def __len__(self) -> int:
            return len(self._modules)

        def __dir__(self):
            keys = super(SequentialContainer, self).__dir__()
            keys = [key for key in keys if not key.isdigit()]
            return keys

        def __iter__(self) -> Iterator[T]:
            return iter(self._modules.values())

        def forward(self, input):
            for module in self:
                input = module(input)
            return input

    return SequentialContainer


def get_list(T):
    class ListContainer(T):
        def __init__(self, modules: Optional[Iterable[T]] = None) -> None:
            super(ListContainer, self).__init__()
            if modules is not None:
                self += modules

        def _get_abs_string_index(self, idx):
            """Get the absolute index for the list of modules"""
            idx = operator.index(idx)
            if not -len(self) <= idx < len(self):
                raise IndexError("index {} is out of range".format(idx))
            if idx < 0:
                idx += len(self)
            return str(idx)

        def __getitem__(self, idx: int) -> T:
            if isinstance(idx, slice):
                return self.__class__(list(self._modules.values())[idx])
            else:
                return self._modules[self._get_abs_string_index(idx)]

        def __setitem__(self, idx: int, module: T) -> None:
            idx = self._get_abs_string_index(idx)
            return setattr(self, str(idx), module)

        def __delitem__(self, idx: Union[int, slice]) -> None:
            if isinstance(idx, slice):
                for k in range(len(self._modules))[idx]:
                    delattr(self, str(k))
            else:
                delattr(self, self._get_abs_string_index(idx))
            str_indices = [str(i) for i in range(len(self._modules))]
            self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

        def __len__(self) -> int:
            return len(self._modules)

        def __iter__(self) -> Iterator[T]:
            return iter(self._modules.values())

        def __iadd__(self: T, modules: Iterable[T]) -> T:
            return self.extend(modules)

        def __dir__(self):
            keys = super(ListContainer, self).__dir__()
            keys = [key for key in keys if not key.isdigit()]
            return keys

        def insert(self, index: int, module: T) -> None:
            """Insert a given module before a given index in the list.
    
            Arguments:
                index (int): index to insert.
                module (nn.Module): module to insert
            """
            for i in range(len(self._modules), index, -1):
                self._modules[str(i)] = self._modules[str(i - 1)]
            self._modules[str(index)] = module

        def append(self: T, module: T) -> T:
            """Appends a given module to the end of the list.
    
            Arguments:
                module (nn.Module): module to append
            """
            self.add_module(str(len(self)), module)
            return self

        def extend(self: T, modules: Iterable[T]) -> T:
            """Appends modules from a Python iterable to the end of the list.
    
            Arguments:
                modules (iterable): iterable of modules to append
            """
            if not isinstance(modules, collections.abc.Iterable):
                raise TypeError(
                    "ModuleList.extend should be called with an iterable, but got "
                    + type(modules).__name__
                )
            offset = len(self)
            for (i, module) in enumerate(modules):
                self.add_module(str(offset + i), module)
            return self

        def forward(self):
            raise NotImplementedError()

    return ListContainer


def get_dict(T):
    class DictContainer(T):
        def __init__(self, modules: Optional[Mapping[str, T]] = None) -> None:
            super(DictContainer, self).__init__()
            if modules is not None:
                self.update(modules)

        def __getitem__(self, key: str) -> T:
            return self._modules[key]

        def __setitem__(self, key: str, module: T) -> None:
            self.add_module(key, module)

        def __delitem__(self, key: str) -> None:
            del self._modules[key]

        def __len__(self) -> int:
            return len(self._modules)

        def __iter__(self) -> Iterator[str]:
            return iter(self._modules)

        def __contains__(self, key: str) -> bool:
            return key in self._modules

        def clear(self) -> None:
            """Remove all items from the ModuleDict.
            """
            self._modules.clear()

        def pop(self, key: str) -> T:
            """Remove key from the ModuleDict and return its module.
    
            Arguments:
                key (string): key to pop from the ModuleDict
            """
            v = self[key]
            del self[key]
            return v

        def keys(self) -> Iterable[str]:
            """Return an iterable of the ModuleDict keys.
            """
            return self._modules.keys()

        def items(self) -> Iterable[Tuple[str, T]]:
            """Return an iterable of the ModuleDict key/value pairs.
            """
            return self._modules.items()

        def values(self) -> Iterable[T]:
            """Return an iterable of the ModuleDict values.
            """
            return self._modules.values()

        def update(self, modules: Mapping[str, T]) -> None:
            if not isinstance(modules, collections.abc.Iterable):
                raise TypeError(
                    "ModuleDict.update should be called with an iterable of key/value pairs, but got "
                    + type(modules).__name__
                )
            if isinstance(modules, (OrderedDict, T, collections.abc.Mapping)):
                for (key, module) in modules.items():
                    self[key] = module
            else:
                for (j, m) in enumerate(modules):
                    if not isinstance(m, collections.abc.Iterable):
                        raise TypeError(
                            "ModuleDict update sequence element #"
                            + str(j)
                            + " should be Iterable; is"
                            + type(m).__name__
                        )
                    if not len(m) == 2:
                        raise ValueError(
                            "ModuleDict update sequence element #"
                            + str(j)
                            + " has length "
                            + str(len(m))
                            + "; 2 is required"
                        )
                    self[m[0]] = m[1]

    return DictContainer


def get_para_list(T):
    class ParameterListContainer(T):
        def __init__(self, parameters=None) -> None:
            super(ParameterListContainer, self).__init__()
            self._initialized = True
            if parameters is not None:
                self += parameters

        def __setstate__(self, state):
            state["_initialized"] = False
            super(ParameterListContainer, self).__setstate__(state)
            self._initialized = True

        def _get_abs_string_index(self, idx):
            """Get the absolute index for the list of modules"""
            idx = operator.index(idx)
            if not -len(self) <= idx < len(self):
                raise IndexError("index {} is out of range".format(idx))
            if idx < 0:
                idx += len(self)
            return str(idx)

        @overload
        def __getitem__(self, idx: int):
            ...

        @overload
        def __getitem__(self: T, idx: slice) -> T:
            ...

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return self.__class__(list(self._parameters.values())[idx])
            else:
                idx = self._get_abs_string_index(idx)
                return self._parameters[str(idx)]

        def __setitem__(self, idx: int, param) -> None:
            idx = self._get_abs_string_index(idx)
            return self.register_parameter(str(idx), param)

        def __len__(self) -> int:
            return len(self._parameters)

        def __iter__(self):
            return iter(self._parameters.values())

        def __iadd__(self, parameters):
            return self.extend(parameters)

        def __dir__(self):
            keys = super(ParameterListContainer, self).__dir__()
            keys = [key for key in keys if not key.isdigit()]
            return keys

        def append(self: T, parameter) -> T:
            """Appends a given parameter at the end of the list.
    
            Arguments:
    
                parameter (nn.Parameter): parameter to append
            """
            self.register_parameter(str(len(self)), parameter)
            return self

        def extend(self: T, parameters) -> T:
            """Appends parameters from a Python iterable to the end of the list.
    
            Arguments:
    
                parameters (iterable): iterable of parameters to append
            """
            if not isinstance(parameters, collections.abc.Iterable):
                raise TypeError(
                    "ParameterList.extend should be called with an iterable, but got "
                    + type(parameters).__name__
                )
            offset = len(self)
            for (i, param) in enumerate(parameters):
                self.register_parameter(str(offset + i), param)
            return self

        def extra_repr(self) -> str:
            child_lines = []
            for (k, p) in self._parameters.items():
                size_str = "x".join((str(size) for size in p.size()))
                device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
                parastr = "Parameter containing: [{} of size {}{}]".format(
                    type(p), size_str, device_str
                )
                child_lines.append("  (" + str(k) + "): " + parastr)
            tmpstr = "\n".join(child_lines)
            return tmpstr

        def __call__(self, input):
            raise RuntimeError("ParameterList should not be called.")

        def _replicate_for_data_parallel(self):
            warnings.warn(
                "nn.ParameterList is being used with DataParallel but this is not supported. This list will appear empty for the models replicated on each GPU except the original one."
            )
            return super(ParameterListContainer, self)._replicate_for_data_parallel()

    return ParameterListContainer


def get_para_dict(T):
    class ParameterDictContainer(T):
        def __init__(self, parameters=None) -> None:
            super(ParameterDictContainer, self).__init__()
            self._initialized = True
            if parameters is not None:
                self.update(parameters)

        def __setstate__(self, state):
            state["_initialized"] = False
            super(ParameterDictContainer, self).__setstate__(state)
            self._initialized = True

        def __getitem__(self, key: str):
            return self._parameters[key]

        def __setitem__(self, key: str, parameter) -> None:
            self.register_parameter(key, parameter)

        def __delitem__(self, key: str) -> None:
            del self._parameters[key]

        def __len__(self) -> int:
            return len(self._parameters)

        def __iter__(self) -> Iterator[str]:
            return iter(self._parameters.keys())

        def __contains__(self, key: str) -> bool:
            return key in self._parameters

        def clear(self) -> None:
            """Remove all items from the ParameterDict.
            """
            self._parameters.clear()

        def pop(self, key: str):
            r"""Remove key from the ParameterDict and return its parameter.
    
            Args:
    
                key (string): key to pop from the ParameterDict
            """
            v = self[key]
            del self[key]
            return v

        def keys(self) -> Iterable[str]:
            r"""Return an iterable of the ParameterDict keys.
            """
            return self._parameters.keys()

        def items(self):
            r"""Return an iterable of the ParameterDict key/value pairs.
            """
            return self._parameters.items()

        def values(self):
            r"""Return an iterable of the ParameterDict values.
            """
            return self._parameters.values()

        def update(self, parameters) -> None:
            r"""Update the :class:`~flow.nn.ParameterDict` with the key-value pairs from a
            mapping or an iterable, overwriting existing keys.
    
            .. note::
                If :attr:`parameters` is an ``OrderedDict``, a :class:`~flow.nn.ParameterDict`, or
                an iterable of key-value pairs, the order of new elements in it is preserved.
         
            Args:
                parameters (iterable): a mapping (dictionary) from string to
                    :class:`~flow.nn.Parameter`, or an iterable of
                    key-value pairs of type (string, :class:`~flow.nn.Parameter`)
    
            """
            if not isinstance(parameters, container_abcs.Iterable):
                raise TypeError(
                    "ParametersDict.update should be called with an "
                    "iterable of key/value pairs, but got " + type(parameters).__name__
                )

            if isinstance(parameters, (OrderedDict, ParameterDictContainer)):
                for key, parameter in parameters.items():
                    self[key] = parameter
            elif isinstance(parameters, container_abcs.Mapping):
                for key, parameter in sorted(parameters.items()):
                    self[key] = parameter
            else:
                for j, p in enumerate(parameters):
                    if not isinstance(p, container_abcs.Iterable):
                        raise TypeError(
                            "ParameterDict update sequence element "
                            "#" + str(j) + " should be Iterable; is" + type(p).__name__
                        )
                    if not len(p) == 2:
                        raise ValueError(
                            "ParameterDict update sequence element "
                            "#"
                            + str(j)
                            + " has length "
                            + str(len(p))
                            + "; 2 is required"
                        )
                    # parameters as length-2 list too cumbersome to type, see ModuleDict.update comment
                    self[p[0]] = p[1]  # type: ignore[assignment]

        def extra_repr(self) -> str:
            child_lines = []
            for k, p in self._parameters.items():
                size_str = "x".join(str(size) for size in p.size())
                device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
                parastr = "Parameter containing: [{} of size {}{}]".format(
                    type(p), size_str, device_str
                )
                child_lines.append("  (" + k + "): " + parastr)
            tmpstr = "\n".join(child_lines)
            return tmpstr

        def __call__(self, input):
            raise RuntimeError("ParameterDict should not be called.")

        def _replicate_for_data_parallel(self):
            warnings.warn(
                "nn.ParameterDict is being used with DataParallel but this is not "
                "supported. This dict will appear empty for the models replicated "
                "on each GPU except the original one."
            )

            return super(ParameterDictContainer, self)._replicate_for_data_parallel()

    return ParameterDictContainer
