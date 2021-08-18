# Nodes represent a definition of a value in oneflow graph of operators.
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
import oneflow as flow
import types
import builtins
if TYPE_CHECKING:
    from .graph import Graph

BaseArgumentTypes = Union[str, int, float, bool, flow.dtype, flow.Tensor, flow.device]
base_types = BaseArgumentTypes.__args__  # type: ignore[attr-defined]

Target = Union[Callable[..., Any], str]

Argument = Optional[Union[
    Tuple[Any, ...],  # actually Argument, but mypy can't represent recursive types
    List[Any],  # actually Argument
    Dict[str, Any],  # actually Argument
    slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
    'Node',
    BaseArgumentTypes
]]

def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [flow, flow.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')


# Borrowed from CPython typing module
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # HACK: In Python 3.6, type aliases from ``typing`` are instances of ``type``, but in
    # later Python versions, type aliases are not instances of ``type``!! We want
    # all type aliases to fall through to ``repr``, so if we have a type that is
    # in the module typing, don't go down this path.
    if isinstance(obj, type) and obj.__module__ != 'typing':
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return('...')
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


def _get_qualified_name(func: Callable[..., Any]) -> str:
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('flow._oneflow_internal.F', 'flow')  # WAR for bug in how torch.ops assigns module
    print(f'{module}.{name}')
    return f'{module}.{name}'

def _format_arg(arg) -> str:
    if isinstance(arg, list):
        items = ', '.join(_format_arg(a) for a in arg)
        return f'[{items}]'
    elif isinstance(arg, tuple):
        items = ', '.join(_format_arg(a) for a in arg)
        maybe_comma = ',' if len(arg) == 1 else ''
        return f'({items}{maybe_comma})'
    elif isinstance(arg, dict):
        items_str = ', '.join(f'{k}: {_format_arg(v)}' for k, v in arg.items())
        return f'{{{items_str}}}'

    if isinstance(arg, Node):
        return '%' + str(arg)
    else:
        return str(arg)

class Node:
    """
    
    The documentation is referenced from: https://pytorch.org/docs/stable/_modules/torch/fx/node.html#Node

    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *including the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """
    def __init__(self, graph: 'Graph', name: str, op: str, target: 'Target',
                 args: Tuple['Argument', ...], kwargs: Dict[str, 'Argument'],
                 type : Optional[Any] = None) -> None:
        self.graph = graph
        self.name = name  # unique name of value being created
        assert op in ['placeholder', 'call_method', 'call_module', 'call_function', 'get_attr', 'output', 'root']
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
        if op in ['call_method', 'call_module']:
            assert isinstance(target, str)
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or torch.add

        # All `Node`-valued inputs. Key is the Node, value is don't-care.
        # The public API for this is `all_input_nodes`, this private attribute
        # should not be accessed directly.
        self._input_nodes : Dict[Node, None] = {}
        self.__update_args_kwargs(map_arg(args, lambda x: x), map_arg(kwargs, lambda x: x))  # type: ignore[arg-type]

        # All of the nodes that use the value produced by this Node
        # Note one user may correspond to several uses, e.g. the node fo ``x + x``
        # would appear once here, but represents two uses.
        #
        # Is a dict to act as an "ordered set". Keys are significant, value dont-care
        self.users : Dict['Node', None] = {}
        # Type expression representing the output value of this node.
        # This should contain the same class of Type objects that would appear
        # as type annotations for function inputs/outputs.
        #
        # For placeholder nodes, this value will be used to type-annotate the
        # generated function parameters.
        # For the return node, this value will be used to type-annotate the
        # generated function return type. (Note this is a special case. ``return``
        # does not produce a value, it's more of a notation. Thus, this value
        # describes the type of args[0] in the ``return`` node.
        self.type : Optional[Any] = type
        self._prev = self
        self._next = self
        self._erased = False

        # If set, use this fn to print this node
        self._repr_fn : Optional[Callable[[Node], str]] = None
        self._stack_trace : Optional[str] = None

        # Dictionary to store metadata passes need to do their
        # transformations. This metadata is preserved across node copies
        self.meta : Dict[str, Any] = {}

    @property
    def next(self) -> 'Node':
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next
    
    @property
    def prev(self) -> 'Node':
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev
    
    def prepend(self, x: 'Node') -> None:
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """
        assert self.graph == x.graph, "Attempting to move a Node into a different Graph"
        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x
    
    def append(self, x: 'Node') -> None:
        """
        Insert x after this node in the list of nodes in the graph.
        Equvalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """
        self._next.prepend(x)
    
    def _remove_from_list(self):
        """
        Delete current node.
        """
        p, n = self._prev, self._next
        p._next, n._prev = n, p
    
    @property
    def args(self) -> Tuple[Argument, ...]:
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._args
    
    @args.setter
    def args(self, a : Tuple[Argument, ...]):
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `__update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.args = new_args`
        self.__update_args_kwargs(map_arg(a, lambda x: x), self._kwargs)  # type: ignore[arg-type]
    
    @property
    def kwargs(self) -> Dict[str, Argument]:
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k : Dict[str, Argument]):
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `__update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.kwargs = new_kwargs`
        self.__update_args_kwargs(self._args, map_arg(k, lambda x: x))  # type: ignore[arg-type]

    @property
    def all_input_nodes(self) -> List['Node']:
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
        return list(self._input_nodes.keys())

    def update_arg(self, idx : int, arg : Argument) -> None:
        """
        Update an existing positional argument to contain the new value
        ``arg``. After calling, ``self.args[idx] == arg``.

        Args:

            idx (int): The index into ``self.args`` of the element to update
            arg (Argument): The new argument value to write into ``args``
        """
        args = list(self.args)
        args[idx] = arg
        self.args = tuple(args)
    
    def update_kwarg(self, key : str, arg : Argument) -> None:
        """
        Update an existing keyword argument to contain the new value
        ``arg``. After calling, ``self.kwargs[key] == arg``.

        Args:

            key (str): The key in ``self.kwargs`` of the element to update
            arg (Argument): The new argument value to write into ``kwargs``
        """
        kwargs = dict(self.kwargs)
        kwargs[key] = arg
        self.kwargs = kwargs

    @property
    def stack_trace(self) -> Optional[str]:
        """
        Return the Python stack trace that was recorded during tracing, if any.
        This property is usually populated by `Tracer.create_proxy`. To record
        stack traces during tracing for debug purposes, set
        `record_stack_traces = True` on the `Tracer` instance.
        """
        return self._stack_trace
    
    @stack_trace.setter
    def stack_trace(self, trace : Optional[str]):
        self._stack_trace = trace
    
    

# _get_qualified_name(flow._oneflow_internal.F.exp)


