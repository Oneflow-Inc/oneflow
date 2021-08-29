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

# TODO(bbuf) Calls patch_function in order to intercept functions at the C-extension level
# Generally, oneflow._C.xxx will not be called when building a Module

class PHBase(object):
    """
    Object representing an input placeholder to `concrete_args`
    """
    def __repr__(self):
        return 'PH'

PH = PHBase()


class Tracer(TracerBase):
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `math`s path from the
    # build environment (e.g. `<module 'math' from '/leaked/path').

    """Tracer(autowrap_modules=(math,), autowrap_functions=(), enable_cpatching=False)

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``oneflow.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """
    def __init__(self, autowrap_modules: Tuple[ModuleType] = (math, ),
                 autowrap_functions: Tuple[Callable, ...] = (),
                 enable_cpatching: bool = False,
                 param_shapes_constant: bool = False) -> None:
        # This method's signature is overridden by the first line of this class'
        # docstring. If this method's signature is modified, the signature that
        # overrides it also should be modified accordingly.

        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap().

            autowrap_function (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap().

            enable_cpatching (bool): defaults to `False`,
                Allows you to enable/disable monkeypatching of oneflow functions at the
                C-level (which captures functins like randn).

                C-level monkeypatching works by directly modifying the PyCFunctionObject*
                so that calling it returns a different function.

                Turning this on is likely to slow down tracing by 1.5-3x.

            param_shapes_constant (bool): see https://github.com/pytorch/pytorch/issues/61733. When
            this flag is set,  calls to shape, size and a few other shape like attributes of a module's parameter
            will be evaluted directly, rather than returning a new Proxy value for an attribute access.

        """

        super().__init__()

        # Functions we will eagerly wrap when we see them while tracing
        # this captures both `math.sqrt()` and `from math import sqrt` automatically
        self._autowrap_function_ids: Set[int] = {
            id(value) for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)}
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))

        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)
        self.enable_cpatching = enable_cpatching
        self.param_shapes_constant = param_shapes_constant

        self.submodule_paths: Optional[Dict[flow.nn.Module, str]] = None
    
    def create_arg(self, a: Any) -> 'Argument':
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_arg`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:

            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.

        This method can be overridden to support more types.

        Args:

            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.


        Returns:

            The value ``a`` converted into the appropriate ``Argument``
        """
        # The base tracer is used to construct Graphs when there is no associated
        # module hierarchy, so it can never create parameter references.
        # The default tracer adds the ability to refer to parameters when
        # tracing modules.
        if isinstance(a, flow.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            raise NameError('parameter is not a member of this module')
        elif isinstance(a, flow.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        elif isinstance(a, flow.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node('call_function', a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.

        if type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            i = 0
            while True:
                qualname = f'_{a.__class__.__name__}_constant_{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})

        return super().create_arg(a)

    def is_leaf_module(self, m: flow.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the OneFlow standard library namespace (flow.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        return m.__module__.startswith('flow.nn') and not isinstance(m, flow.nn.Sequential)
    
    def path_of_module(self, mod : flow.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # Prefer the O(1) algorithm
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError('module is not installed as a submodule')
            assert isinstance(path, str)
            return path
        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError('module is not installed as a submodule')
