import builtins
import functools
import inspect
import math
import os
from types import CodeType, FunctionType, ModuleType
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, Type, List, Callable, Union
from itertools import chain
import oneflow as flow
import oneflow.fx.utils._pytree as pytree

import sys
from .node import Argument, map_aggregate, base_types
from .graph import Graph, _PyTreeInfo
from .graph_module import GraphModule
from .proxy import TracerBase, Proxy, ParameterProxy

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

# These need to run in global scope to handle nested calls correctly
_orig_module_call : Callable = flow.nn.Module.__call__
_orig_module_getattr : Callable = flow.nn.Module.__getattr__


_proxyable_classes : Dict[Type, None] = {}

class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import oneflow
        import oneflow.fx

        class TensorPair(metaclass=oneflow.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_ctor(x : TensorPair, y : oneflow.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)

        x = TensorPair(oneflow.randn(5, 3), oneflow.randn(5, 3))
        y = oneflow.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = oneflow.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : oneflow.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that contruction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """
    def __init__(cls, name, bases, attrs):
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)  # type: ignore[call-overload]

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        map_aggregate(args, check_proxy)
        map_aggregate(kwargs, check_proxy)

        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy('call_function', cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance

def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args : tuple
    if hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs, 0,
            0, co.co_nlocals, co.co_stacksize,
            co_flags, co.co_code, co.co_consts, co.co_names,
            co.co_varnames, co.co_filename, co.co_name,
            co.co_firstlineno, co.co_lnotab, co.co_freevars,
            co.co_cellvars
        )
    else:
        co_args = (
            nargs, 0, co.co_nlocals,
            co.co_stacksize, co_flags, co.co_code, co.co_consts,
            co.co_names, co.co_varnames, co.co_filename,
            co.co_name, co.co_firstlineno, co.co_lnotab,
            co.co_freevars, co.co_cellvars)
    new_code = CodeType(*co_args)  # type: ignore[arg-type]
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)

    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables

class PHBase(object):
    """
    Object representing an input placeholder to `concrete_args`
    """
    def __repr__(self):
        return 'PH'

PH = PHBase()


