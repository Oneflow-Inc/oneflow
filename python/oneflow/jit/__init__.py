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
import warnings
from typing import Any, Dict, List, Set, Tuple, Union, Callable

warnings.warn(
    "The oneflow.jit interface is just to align the torch.jit interface and has no practical significance."
)

def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    warnings.warn(
        "The oneflow.jit.script interface is just to align the torch.jit.script interface and has no practical significance."
    )
    return obj


def ignore(drop=False, **kwargs):
    warnings.warn(
        "The oneflow.jit.ignore interface is just to align the torch.jit.ignore interface and has no practical significance."
    )

    def decorator(fn):
        return fn

    return decorator


def unused(fn):
    warnings.warn(
        "The oneflow.jit.unused interface is just to align the torch.jit.unused interface and has no practical significance."
    )

    return fn


def _script_if_tracing(fn):
    warnings.warn(
        "The oneflow.jit._script_if_tracing interface is just to align the torch.jit._script_if_tracing interface and has no practical significance."
    )

    return fn


def _overload_method(fn):
    warnings.warn(
        "The oneflow.jit._overload_method interface is just to align the torch.jit._overload_method interface and has no practical significance."
    )

    return fn


def is_scripting():
    return False


def is_tracing():
    return False

class _Final:
    """Mixin to prohibit subclassing"""

    __slots__ = ('__weakref__',)

    def __init_subclass__(self, *args, **kwds):
        if '_root' not in kwds:
            raise TypeError("Cannot subclass special typing classes")

class _SpecialForm(_Final, _root=True):
    __slots__ = ('_name', '__doc__', '_getitem')

    def __init__(self, getitem):
        self._getitem = getitem
        self._name = getitem.__name__
        self.__doc__ = getitem.__doc__

    def __getattr__(self, item):
        if item in {'__name__', '__qualname__'}:
            return self._name

        raise AttributeError(item)

    def __mro_entries__(self, bases):
        raise TypeError(f"Cannot subclass {self!r}")

    def __repr__(self):
        return 'typing.' + self._name

    def __reduce__(self):
        return self._name

    def __call__(self, *args, **kwds):
        raise TypeError(f"Cannot instantiate {self!r}")

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __instancecheck__(self, obj):
        raise TypeError(f"{self} cannot be used with isinstance()")

    def __subclasscheck__(self, cls):
        raise TypeError(f"{self} cannot be used with issubclass()")

    def __getitem__(self, parameters):
        return self._getitem(self, parameters)

@_SpecialForm
def Final(*args, **kwargs):
    warnings.warn(
        "The oneflow.jit.Final interface is just to align the torch.jit.Final interface and has no practical significance."
    )

def interface(fn):
    warnings.warn(
        "The oneflow.jit.interface interface is just to align the torch.jit.interface interface and has no practical significance."
    )
    return fn

