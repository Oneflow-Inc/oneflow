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
)

import oneflow as flow
from oneflow.nn.module import Module

T = TypeVar("T")


class Sequential(Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example:

    .. code-block:: python

        >>> import oneflow.nn as nn
        >>> nn.Sequential(nn.Conv2d(1,20,5), nn.ReLU(), nn.Conv2d(20,64,5), nn.ReLU()) #doctest: +ELLIPSIS
        Sequential(
          (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
          (1): ReLU()
          (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
          (3): ReLU()
        )
        >>> nn.Sequential(OrderedDict([
        ...    ('conv1', nn.Conv2d(1,20,5)),
        ...    ('relu1', nn.ReLU()),
        ...    ('conv2', nn.Conv2d(20,64,5)),
        ...    ('relu2', nn.ReLU())
        ... ])) #doctest: +ELLIPSIS
        Sequential(
          (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
          (relu1): ReLU()
          (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
          (relu2): ReLU()
        )

    """

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Module]") -> None:
        ...

    def __init__(self, *args: Any):
        super(Sequential, self).__init__()
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

    def __setitem__(self, idx: int, module: Module) -> None:
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
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class ModuleList(Module):
    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super(ModuleList, self).__init__()
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

    def __getitem__(self, idx: int) -> Module:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
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

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self: T, modules: Iterable[Module]) -> T:
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        """Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self: T, module: Module) -> T:
        """Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self: T, modules: Iterable[Module]) -> T:
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


class ModuleDict(Module):
    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
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

    def pop(self, key: str) -> Module:
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

    def items(self) -> Iterable[Tuple[str, Module]]:
        """Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        """Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        if not isinstance(modules, collections.abc.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an iterable of key/value pairs, but got "
                + type(modules).__name__
            )
        if isinstance(modules, (OrderedDict, ModuleDict, collections.abc.Mapping)):
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


class ParameterList(Module):
    def __init__(self, parameters: Optional[Iterable["Parameter"]] = None) -> None:
        super(ParameterList, self).__init__()
        self._initialized = True
        if parameters is not None:
            self += parameters

    def __setstate__(self, state):
        state["_initialized"] = False
        super(ParameterList, self).__setstate__(state)
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
    def __getitem__(self, idx: int) -> "Parameter":
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

    def __setitem__(self, idx: int, param: "Parameter") -> None:
        idx = self._get_abs_string_index(idx)
        return self.register_parameter(str(idx), param)

    def __setattr__(self, key: Any, value: Any) -> None:
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and (not isinstance(value, flow.nn.Parameter)):
                warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator["Parameter"]:
        return iter(self._parameters.values())

    def __iadd__(self: T, parameters: Iterable["Parameter"]) -> T:
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self: T, parameter: "Parameter") -> T:
        """Appends a given parameter at the end of the list.

        Arguments:

            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self: T, parameters: Iterable["Parameter"]) -> T:
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
        return super(ParameterList, self)._replicate_for_data_parallel()


class ParameterDict(Module):
    def __init__(self, parameters: Optional[Mapping[str, "Parameter"]] = None) -> None:
        super(ParameterDict, self).__init__()
        self._initialized = True
        if parameters is not None:
            self.update(parameters)

    def __setstate__(self, state):
        state["_initialized"] = False
        super(ParameterDict, self).__setstate__(state)
        self._initialized = True

    def __getitem__(self, key: str) -> "Parameter":
        return self._parameters[key]

    def __setitem__(self, key: str, parameter: "Parameter") -> None:
        self.register_parameter(key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and (not isinstance(value, flow.nn.Parameter)):
                warnings.warn("Setting attributes on ParameterDict is not supported.")
        super(ParameterDict, self).__setattr__(key, value)

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

    def pop(self, key: str) -> "Parameter":
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

    def items(self) -> Iterable[Tuple[str, "Parameter"]]:
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()

    def values(self) -> Iterable["Parameter"]:
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()

    def update(self, parameters: Mapping[str, "Parameter"]) -> None:
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

        if isinstance(parameters, (OrderedDict, ParameterDict)):
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
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
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

        return super(ParameterDict, self)._replicate_for_data_parallel()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
