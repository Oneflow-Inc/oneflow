.. currentmodule:: oneflow.fx

oneflow.fx
=============

Overview
--------

FX is a toolkit for developers to use to transform ``nn.Module``
instances. FX consists of three main components: a **symbolic tracer,**
an **intermediate representation**, and **Python code generation**. A
demonstration of these components in action:

::

    import oneflow
    
    # Simple module for demonstration
    
    class MyModule(oneflow.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = oneflow.nn.Parameter(oneflow.rand(3, 4))
            self.linear = oneflow.nn.Linear(4, 5)

        def forward(self, x):
            return self.linear(x + self.param).clamp(min=0.0, max=1.0)

    module = MyModule()

    from oneflow.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced : oneflow.fx.GraphModule = symbolic_trace(module)

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)
    
    """
    
    graph():
        %x : [#users=1] = placeholder[target=x]
        %param : [#users=1] = get_attr[target=param]
        %add : [#users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
        %linear : [#users=1] = call_module[target=linear](args = (%add,), kwargs = {})
        %clamp : [#users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
        return clamp

    """

    # Code generation - valid Python code
    print(symbolic_traced.code)
    """

    def forward(self, x):
        param = self.param
        add = x + param;  x = param = None
        linear = self.linear(add);  add = None
        clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
        return clamp

    """

The **symbolic tracer** performs "symbolic execution" of the Python
code. It feeds fake values, called Proxies, through the code. Operations
on theses Proxies are recorded. More information about symbolic tracing
can be found in the :func:`symbolic_trace` and :class:`Tracer`
documentation.

The **intermediate representation** is the container for the operations
that were recorded during symbolic tracing. It consists of a list of
Nodes that represent function inputs, callsites (to functions, methods,
or :class:`oneflow.nn.Module` instances), and return values. More information
about the IR can be found in the documentation for :class:`Graph`. The
IR is the format on which transformations are applied.

**Python code generation** is what makes FX a Python-to-Python (or
Module-to-Module) transformation toolkit. For each Graph IR, we can
create valid Python code matching the Graph's semantics. This
functionality is wrapped up in :class:`GraphModule`, which is a
:class:`oneflow.nn.Module` instance that holds a :class:`Graph` as well as a
``forward`` method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing ->
intermediate representation -> transforms -> Python code generation)
constitutes the Python-to-Python transformation pipeline of FX. In
addition, these components can be used separately. For example,
symbolic tracing can be used in isolation to capture a form of
the code for analysis (and not transformation) purposes. Code
generation can be used for programmatically generating models, for
example from a config file. There are many uses for FX!

Several example transformations can be found at the
`examples <https://github.com/Oneflow-Inc/examples/tree/main/fx>`__
repository.

.. _Writing Transformations:


Writing Transformations
-----------------------
What is an FX transform? Essentially, it's a function that looks like this.

::

    import oneflow as flow

    def transform(m: flow.nn.Module,
                    tracer_class : type = flow.fx.Tracer) -> flow.nn.Module:
        # Step 1: Acquire a Graph representing the code in `m`

        # NOTE: flow.fx.symbolic_trace is a wrapper around a call to
        # fx.Tracer.trace and constructing a GraphModule. We'll
        # split that out in our transform to allow the caller to
        # customize tracing behavior.
        graph : flow.fx.Graph = tracer_class().trace(m)

        # Step 2: Modify this Graph or create a new one
        graph = ...

        # Step 3: Construct a Module to return
        return flow.fx.GraphModule(m, graph)


Your transform will take in an :class:`oneflow.nn.Module`, acquire a :class:`Graph`
from it, do some modifications, and return a new :class:`oneflow.nn.Module`. You 
should think of the :class:`oneflow.nn.Module` that your FX transform returns as 
identical to a regular :class:`oneflow.nn.Module` -- you can pass it to another
FX transform or other IR in oneflow, or you can run it. Ensuring that the inputs 
and outputs of your FX transform are a :class:`oneflow.nn.Module` will allow for composability.

.. note::

    It is also possible to modify an existing :class:`GraphModule` instead of
    creating a new one, like so::

        import oneflow as flow

        def transform(m : nn.Module) -> nn.Module:
            gm : flow.fx.GraphModule = flow.fx.symbolic_trace(m)

            # Modify gm.graph
            # <...>

            # Recompile the forward() method of `gm` from its Graph
            gm.recompile()

            return gm

    Note that you MUST call :meth:`GraphModule.recompile` to bring the generated
    ``forward()`` method on the ``GraphModule`` in sync with the modified :class:`Graph`.

Given that you’ve passed in a :class:`oneflow.nn.Module` that has been traced into a
:class:`Graph`, there are now two primary approaches you can take to building a new
:class:`Graph`.

A Quick Primer on Graphs
^^^^^^^^^^^^^^^^^^^^^^^^

Full treatment of the semantics of graphs can be found in the :class:`Graph`
documentation, but we are going to cover the basics here. A :class:`Graph` is
a data structure that represents a method on a :class:`GraphModule`. The
information that this requires is:

- What are the inputs to the method?
- What are the operations that run inside the method?
- What is the output (i.e. return) value from the method?

All three of these concepts are represented with :class:`Node` instances.
Let's see what we mean by that with a short example:

::

    import oneflow as flow

    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = flow.nn.Parameter(flow.rand(3, 4))
            self.linear = flow.nn.Linear(4, 5)

        def forward(self, x):
            return flow.topk(flow.sum(
                self.linear(x + self.linear.weight).relu(), dim=-1), 3)

    m = MyModule()
    gm = flow.fx.symbolic_trace(m)

    gm.graph.print_tabular()

Here we define a module ``MyModule`` for demonstration purposes, instantiate it,
symbolically trace it, then call the :meth:`Graph.print_tabular` method to print
out a table showing the nodes of this :class:`Graph`:


    +---------------+---------------+----------------------------+--------------------+-------------+
    | opcode        | name          | target                     | args               | kwargs      |
    +===============+===============+============================+====================+=============+
    | placeholder   | x             | x                          | ()                 | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | get_attr      | linear_weight | linear.weight              | ()                 | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_function | add           | <built-in function add>    | (x, linear_weight) | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_module   | linear        | linear                     | (add  ,)           | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | relu          | relu                       | (linear  ,)        | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | sum_1         | sum                        | (relu,)            | {'dim': -1} |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | topk          | topk                       | (sum_1, 3)         | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | output        | output        | output                     | (topk,)            | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+


We can use this information to answer the questions we posed above.

- What are the inputs to the method? In FX, method inputs are specified
  via special ``placeholder`` nodes. In this case, we have a single
  ``placeholder`` node with a ``target`` of ``x``, meaning we have
  a single (non-self) argument named x.
- What are the operations within the method? The ``get_attr``,
  ``call_function``, ``call_module``, and ``call_method`` nodes
  represent the operations in the method. A full treatment of
  the semantics of all of these can be found in the :class:`Node`
  documentation.
- What is the return value of the method? The return value in a
  :class:`Graph` is specified by a special ``output`` node.

Given that we now know the basics of how code is represented in
FX, we can now explore how we would edit a :class:`Graph`.

Graph Manipulation
^^^^^^^^^^^^^^^^^^

Direct Graph Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~

One approach to building this new :class:`Graph` is to directly manipulate your old
one. To aid in this, we can simply take the :class:`Graph` we obtain from symbolic
tracing and modify it. For example, let’s say we desire to replace `Tensor.add` calls with `Tensor.mul` calls.


::

    import oneflow as flow

    # Sample module
    class M(flow.nn.Module):
        def forward(self, x, y):
            return x.add(y)

    def transform(m: flow.nn.Module,
                    tracer_class : type = flow.fx.Tracer) -> flow.fx.GraphModule:
        
        graph : flow.fx.Graph = tracer_class().trace(m)

        # FX represents its Graph as an ordered list of
        # nodes, so we can iterate through them.

        for node in graph.nodes:
            
            # Checks if we're calling a method
            #  (i.e:
            # flow.add)
            
            if node.op == 'call_method':
                
                # The target attribute is the method
                # that call_method calls.
                
                if hasattr(flow, 'add'):
                    node.target = 'mul'

        graph.lint() # Does some checks to make sure the
                        # Graph is well-formed.

        return flow.fx.GraphModule(m, graph)

    m = M()
    new_m = transform(m)
    print(new_m.graph)
    """
    graph():
        %x : [#users=1] = placeholder[target=x]
        %y : [#users=1] = placeholder[target=y]
        %add : [#users=1] = call_method[target=mul](args = (%x, %y), kwargs = {})
        return add
    """


We can also do more involved :class:`Graph` rewrites, such as
deleting or appending nodes. To aid in these transformations,
FX has utility functions for transforming the graph that can
be found in the :class:`Graph` documentation. An
example of using these APIs to append a :func:`flow.relu` call
can be found below.

::

    # Specifies the insertion point. Any nodes added to the
    # Graph within this scope will be inserted after `node`
    with traced.graph.inserting_after(node):
        # Insert a new `call_function` node calling `flow.relu`
        new_node = traced.graph.call_function(
            flow.relu, args=(node,))

        # We want all places that used the value of `node` to
        # now use that value after the `relu` call we've added.
        # We use the `replace_all_uses_with` API to do this.
        node.replace_all_uses_with(new_node)

For simple transformations that only consist of substitutions, you can also
make use of the `subgraph rewriter. <https://github.com/Oneflow-Inc/oneflow/tree/master/python/oneflow/fx/subgraph_rewriter.py>`__

Subgraph Rewriting With replace_pattern()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FX also provides another level of automation on top of direct graph manipulation.
The :func:`replace_pattern` API is essentially a "find/replace" tool for editing
:class:`Graph`\s. It allows you to specify a ``pattern`` and ``replacement`` function
and it will trace through those functions, find instances of the group of operations
in the ``pattern`` graph, and replace those instances with copies of the ``replacement``
graph. This can help to greatly automate tedious graph manipulation code, which can
get unwieldy as the transformations get more complex.


Graph Manipulation Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Replace one
   op <https://github.com/Oneflow-Inc/examples/blob/main/fx/replace_op.py>`__
-  `replace_pattern: Basic usage <https://github.com/Oneflow-Inc/examples/blob/main/fx/subgraph_rewriter_basic_use.py>`__
-  `Invert Transformation <https://github.com/Oneflow-Inc/examples/blob/main/fx/invert.py>`__


Proxy/Retracing
^^^^^^^^^^^^^^^

Another way of manipulating :class:`Graph`\s is by reusing the :class:`Proxy`
machinery used in symbolic tracing. For example, let’s
imagine that we wanted to write a transformation that decomposed
OneFlow functions into smaller operations. It would transform every
``F.relu(x)`` call into ``(x > 0) * x``. One possibility would be to
perform the requisite graph rewriting to insert the comparison and
multiplication after the ``F.relu``, and then clean up the original
``F.relu``. However, we can automate this process by using :class:`Proxy`
objects to automatically record operations into the :class:`Graph`.

To use this method, we write the operations that we want inserted as regular
OneFlow code and invoke that code with :class:`Proxy` objects as arguments.
These :class:`Proxy` objects will capture the operations that are performed
on them and append them to the :class:`Graph`.

::

    import oneflow as flow
    import oneflow.fx as fx
    import oneflow.nn.functional as F

    def relu_decomposition(x):
        return (x > 0) * x

    decomposition_rules = {}
    decomposition_rules[F.relu] = relu_decomposition

    def decompose(model: flow.nn.Module,
                    tracer_class : type = fx.Tracer) -> flow.nn.Module:
        """
        Decompose `model` into smaller constituent operations.
        Currently,this only supports decomposing ReLU into its
        mathematical definition: (x > 0) * x
        """
        graph : fx.Graph = tracer_class().trace(model)
        new_graph = fx.Graph()
        env = {}
        for node in graph.nodes:
            if node.op == 'call_function' and node.target in decomposition_rules:
                # By wrapping the arguments with proxies,
                # we can dispatch to the appropriate
                # decomposition rule and implicitly add it
                # to the Graph by symbolically tracing it.
                proxy_args = [
                    fx.Proxy(env[x.name]) if isinstance(x, fx.Node) else x for x in node.args]
                output_proxy = decomposition_rules[node.target](*proxy_args)

                # Operations on `Proxy` always yield new `Proxy`s, and the
                # return value of our decomposition rule is no exception.
                # We need to extract the underlying `Node` from the `Proxy`
                # to use it in subsequent iterations of this transform.
                new_node = output_proxy.node
                env[node.name] = new_node
            else:
                # Default case: we don't have a decomposition rule for this
                # node, so just copy the node over into the new graph.
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
        return fx.GraphModule(model, new_graph)


The Interpreter Pattern
^^^^^^^^^^^^^^^^^^^^^^^

A useful code organizational pattern in FX is to loop over all the :class:`Node`\s
in a :class:`Graph` and execute them. This can be used for several things including
runtime analysis of values flowing through the graph or transformation of the code
via retracing with :class:`Proxy`\s. For example, suppose we want to run a
:class:`GraphModule` and record the :class:`oneflow.Tensor` shape and dtype
properties on the nodes as we see them at runtime. That might look like:

::

    import oneflow as flow
    from oneflow.fx.node import Node

    from typing import Dict

    class ShapeProp:
        """
        Shape propagation. This class takes a `GraphModule`.
        Then, its `propagate` method executes the `GraphModule`
        node-by-node with the given arguments. As each operation
        executes, the ShapeProp class stores away the shape and
        element type for the output values of each operation on
        the `shape` and `dtype` attributes of the operation's
        `Node`.
        """
        def __init__(self, mod):
            self.mod = mod
            self.graph = mod.graph
            self.modules = dict(self.mod.named_modules())

        def propagate(self, *args):
            args_iter = iter(args)
            env : Dict[str, Node] = {}

            def load_arg(a):
                return flow.fx.graph.map_arg(a, lambda n: env[n.name])

            def fetch_attr(target : str):
                target_atoms = target.split('.')
                attr_itr = self.mod
                for i, atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                return attr_itr

            for node in self.graph.nodes:
                if node.op == 'placeholder':
                    result = next(args_iter)
                elif node.op == 'get_attr':
                    result = fetch_attr(node.target)
                elif node.op == 'call_function':
                    result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                elif node.op == 'call_method':
                    self_obj, *args = load_arg(node.args)
                    kwargs = load_arg(node.kwargs)
                    result = getattr(self_obj, node.target)(*args, **kwargs)
                elif node.op == 'call_module':
                    result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

                # This is the only code specific to shape propagation.
                # you can delete this `if` branch and this becomes
                # a generic GraphModule interpreter.
                if isinstance(result, flow.Tensor):
                    node.shape = result.shape
                    node.dtype = result.dtype

                env[node.name] = result

            return load_arg(self.graph.result)


As you can see, a full interpreter for FX is not that complicated
but it can be very useful. To ease using this pattern, we provide
the :class:`Interpreter` class, which encompasses the above logic
in a way that certain aspects of the interpreter's execution can
be overridden via method overrides.

In addition to executing operations, we can also generate a new
`Graph` by feeding :class:`Proxy` values through an interpreter.
Similarly, we provide the :class:`Transformer` class to encompass
this pattern. :class:`Transformer` behaves similarly to
:class:`Interpreter`, but instead of calling the ``run`` method to
get a concrete output value from the Module, you would call the
:meth:`Transformer.transform` method to return a new
:class:`GraphModule` which was subject to any transformation rules
you installed as overridden methods.


Examples of the Interpreter Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#TODO(BBuf) add examples

Debugging the Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we've identified that a transformation is creating incorrect
code, it's time to debug the transformation itself. First, we'll check
the :ref:`Limitations of Symbolic Tracing` section in the documentation.
Once we verify that tracing is working as expected, the goal
becomes figuring out what went wrong during our ``GraphModule``
transformation. There may be a quick answer in
:ref:`Writing Transformations`, but, if not, there are several ways to
examine our traced module:

::

    # Sample Module
    class M(flow.nn.Module):
        def forward(self, x, y):
            return x + y

    # Create an instance of `M`
    m = M()

    # Symbolically trace an instance of `M` (returns a GraphModule). In
    # this example, we'll only be discussing how to inspect a
    # GraphModule, so we aren't showing any sample transforms for the
    # sake of brevity.
    traced = symbolic_trace(m)

    # Print the code produced by tracing the module.
    print(traced)
    # The generated `forward` function is:
    """
    def forward(self, x, y):
        add = x + y;  x = y = None
        return add
    """

    # Print the internal Graph.
    print(traced.graph)
    # This print-out returns:
    """
    graph():
        %x : [#users=1] = placeholder[target=x]
        %y : [#users=1] = placeholder[target=y]
        %add : [#users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
        return add
    """

    # Print a tabular representation of the internal Graph.
    traced.graph.print_tabular()
    # This gives us:
    """
    opcode         name    target                   args    kwargs
    -------------  ------  -----------------------  ------  --------
    placeholder    x       x                        ()      {}
    placeholder    y       y                        ()      {}
    call_function  add     <built-in function add>  (x, y)  {}
    output         output  output                   (add,)  {}
    """

Using the utility functions above, we can compare our traced Module
before and after we've applied our transformations. Sometimes, a
simple visual comparison is enough to trace down a bug. If it's still
not clear what's going wrong, a debugger like ``pdb`` can be a good
next step.

Going off of the example above, consider the following code:

::

    # Sample user-defined function
    def transform_graph(module: flow.nn.Module, tracer_class : type = fx.Tracer) -> flow.nn.Module:
        # Get the Graph from our traced Module
        g = tracer_class().trace(module)

        """
        Transformations on `g` go here
        """

        return fx.GraphModule(module, g)

    # Transform the Graph
    transformed = transform_graph(traced)

    # Print the new code after our transforms. Check to see if it was
    # what we expected
    print(transformed)

Using the above example, let’s say that the call to ``print(traced)``
showed us that there was an error in our transforms. We want to find
what goes wrong using a debugger. We start a ``pdb`` session. We can see
what’s happening during the transform by breaking on
``transform_graph(traced)``, then pressing ``s`` to “step into” the call
to ``transform_graph(traced)``.

We may also have good luck by editing the ``print_tabular`` method to print
different attributes of the Nodes in the Graph. (For example, we might
want to see the Node’s ``input_nodes`` and ``users``.)

.. _Available Debuggers:

Available Debuggers
^^^^^^^^^^^^^^^^^^^^^^

The most common Python debugger is
`pdb <https://docs.python.org/3/library/pdb.html>`__. You can start
your program in “debug mode” with ``pdb`` by typing
``python -m pdb FILENAME.py`` into the command line, where ``FILENAME``
is the name of the file you want to debug. After that, you can use the
``pdb`` `debugger commands
<https://docs.python.org/3/library/pdb.html#debugger-commands>`__
to move through your running program stepwise. It’s common to set a
breakpoint (``b LINE-NUMBER``) when you start ``pdb``, then call ``c`` to
run the program until that point. This prevents you from having to step
through each line of execution (using ``s`` or ``n``) to get to the part
of the code you want to examine. Alternatively, you can write
``import pdb; pdb.set_trace()`` before the line you want to break at.
If you add ``pdb.set_trace()``, your program will automatically start
in debug mode when you run it. (In other words, you can just type
``python FILENAME.py`` into the command line instead of
``python -m pdb FILENAME.py``.) Once you're running your file in
debug mode, you can step through the code and examine your program's
internal state using certain commands. There are many excellent
tutorials on ``pdb`` online, including RealPython’s
`“Python Debugging With Pdb” <https://realpython.com/python-debugging-pdb/>`__.

IDEs like PyCharm or VSCode usually have a debugger built in. In your
IDE, you can choose to either a) use ``pdb`` by pulling up a terminal
window in your IDE (e.g. View → Terminal in VSCode), or b) use the
built-in debugger (usually a graphical wrapper around ``pdb``).

.. _Limitations of Symbolic Tracing:

Limitations of Symbolic Tracing
-------------------------------

FX uses a system of **symbolic tracing** (a.k.a `symbolic
execution <https://en.wikipedia.org/wiki/Symbolic_execution>`__)
to capture the semantics of programs in a transformable/analyzable form.
The system is **tracing** in that it executes the program (really a
:class:`oneflow.nn.Module` or function) to record operations. It is
**symbolic** in that the data flowing through the program during this
execution is not real data, but rather symbols (:class:`Proxy` in FX parlance).

Although symbolic tracing works for most neural net code, it has some
limitations.

Dynamic Control Flow
^^^^^^^^^^^^^^^^^^^^

The main limitation of symbolic tracing is it does not currently support
*dynamic control flow*. That is, loops or ``if`` statements where the
condition may depend on the input values of the program.

For example, let’s examine the following program:

::

    def func_to_trace(x):
        if x.sum() > 0:
            return flow.relu(x)
        else:
            return flow.neg(x)

    traced = flow.fx.symbolic_trace(func_to_trace)
    """
      <...>
      File "dyn.py", line 6, in func_to_trace
        if x.sum() > 0:
      File "oneflow/python/oneflow/fx/proxy.py", line 279, in __bool__
        return self.tracer.to_bool(self)
      File "oneflow/python/oneflow/fx/proxy.py", line 176, in to_bool
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
    oneflow.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
    """

The condition to the ``if`` statement relies on the value of ``x.sum()``,
which relies on the value of ``x``, a function input. Since
``x`` can change (i.e. if you pass a new input tensor to the traced
function), this is *dynamic control flow*. The traceback walks back up
through your code to show you where this situation happens.

Static Control Flow
~~~~~~~~~~~~~~~~~~~

On the other hand, so-called *static control flow* is supported. Static
control flow is loops or ``if`` statements whose value cannot change
across invocations. Typically, in OneFlow programs, this control flow
arises for code making decisions about a model’s architecture based on
hyper-parameters. As a concrete example:

::

    import oneflow as flow

    class MyModule(flow.nn.Module):
        def __init__(self, do_activation : bool = False):
            super().__init__()
            self.do_activation = do_activation
            self.linear = flow.nn.Linear(512, 512)

        def forward(self, x):
            x = self.linear(x)
            # This if-statement is so-called static control flow.
            # Its condition does not depend on any input values
            if self.do_activation:
                x = flow.relu(x)
            return x

    without_activation = MyModule(do_activation=False)
    with_activation = MyModule(do_activation=True)

    traced_without_activation = flow.fx.symbolic_trace(without_activation)
    print(traced_without_activation.code)
    """
    def forward(self, x):
        linear = self.linear(x);  x = None
        return linear
    """

    traced_with_activation = flow.fx.symbolic_trace(with_activation)
    print(traced_with_activation.code)
    """
    def forward(self, x):
        linear = self.linear(x);  x = None
        relu = oneflow.relu(linear);  linear = None
        return relu
    """

The if-statement ``if self.do_activation`` does not depend on any
function inputs, thus it is static. ``do_activation`` can be considered
to be a hyper-parameter, and the traces of different instances of
``MyModule`` with different values for that parameter have different
code. This is a valid pattern that is supported by symbolic tracing.

Many instances of dynamic control flow are semantically static control
flow. These instances can be made to support symbolic tracing by
removing the data dependencies on input values, for example by moving
values to ``Module`` attributes or by binding concrete values to arguments
during symbolic tracing:

::

        def f(x, flag):
            if flag: return x
            else: return x*2

        fx.symbolic_trace(f) # Fails!

        fx.symbolic_trace(f, concrete_args={'flag': True})

In the case of truly dynamic control flow, the sections of the program
that contain this code can be traced as calls to the Method (see
:ref:`Customizing Tracing`) or function (see
:func:`wrap`) rather than tracing through them.

Non-\ ``oneflow`` Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

FX uses ``__oneflow_function__`` as the mechanism by which it intercepts
calls (see the `technical
overview <https://github.com/Oneflow-Inc/oneflow/blob/master/python/oneflow/fx/OVERVIEW.md#technical-details>`__
for more information about this). Some functions, such as builtin Python
functions or those in the ``math`` module, are not covered by
``__oneflow_function__``, but we would still like to capture them in
symbolic tracing. For example:

::

    import oneflow as flow
    from math import sqrt

    def normalize(x):
        """
        Normalize `x` by the size of the batch dimension
        """
        return x / sqrt(len(x))

    # It's valid Python code
    normalize(flow.rand(3, 4))

    traced = flow.fx.symbolic_trace(normalize)

    """
      <...>
      File "sqrt.py", line 8, in normalize
        return x / sqrt(len(x))
      File "oneflow/python/oneflow/fx/proxy.py", line 285, in __len__
        raise RuntimeError(
    RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call oneflow.fx.wrap('len') at module scope
    """

The error tells us that the built-in function ``len`` is not supported.
We can make it so that functions like this are recorded in the trace as
direct calls using the :func:`wrap` API:

::

    flow.fx.wrap('len')
    flow.fx.wrap('sqrt')

    traced = flow.fx.symbolic_trace(normalize)

    print(traced.code)
    """
    import math
    def forward(self, x):
        len_1 = len(x)
        sqrt = math.sqrt(len_1);  len_1 = None
        truediv = x / sqrt;  x = sqrt = None
        return truediv
    """

.. _Customizing Tracing:

Customizing Tracing with the ``Tracer`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`Tracer` class is the class that underlies the
implementation of ``symbolic_trace``. The behavior of tracing can be
customized by subclassing Tracer, like so:

::

    class MyCustomTracer(flow.fx.Tracer):
        # Inside here you can override various methods
        # to customize tracing. See the `Tracer` API
        # reference
        pass


    # Let's use this custom tracer to trace through this module
    class MyModule(flow.nn.Module):
        def forward(self, x):
            return flow.relu(x) + flow.ones(3, 4)

    mod = MyModule()

    traced_graph = MyCustomTracer().trace(mod)
    # trace() returns a Graph. Let's wrap it up in a
    # GraphModule to make it runnable
    traced = flow.fx.GraphModule(mod, traced_graph)

Leaf Modules
~~~~~~~~~~~~

Leaf Modules are the modules that appear as calls in the symbolic trace
rather than being traced through. The default set of leaf modules is the
set of standard ``oneflow.nn`` module instances. For example:

::

    class MySpecialSubmodule(flow.nn.Module):
        def forward(self, x):
            return flow.negative(x)

    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = flow.nn.Linear(3, 4)
            self.submod = MySpecialSubmodule()

        def forward(self, x):
            return self.submod(self.linear(x))

    traced = oneflow.fx.symbolic_trace(MyModule())
    print(traced.code)
    # `linear` is preserved as a call, yet `submod` is traced though.
    # This is because the default set of "Leaf Modules" includes all
    # standard `oneflow.nn` modules.
    """
    def forward(self, x):
        linear = self.linear(x);  x = None
        negative = linear.negative();  linear = None
        return negative
    """

The set of leaf modules can be customized by overriding
:meth:`Tracer.is_leaf_module`.

Miscellanea
^^^^^^^^^^^

-  Tensor constructors (e.g. ``oneflow.zeros``, ``oneflow.ones``,
   ``oneflow.rand``, ``oneflow.randn``)
   are currently not traceable.

   -  The deterministic constructors (``zeros``, ``ones``) can be used
      and the value they produce will be embedded in the trace as a
      constant. This is only problematic if the arguments to these
      constructors refers to dynamic input sizes. In this case,
      ``ones_like`` or ``zeros_like`` may be a viable substitute.
   -  Nondeterministic constructors (``rand``, ``randn``) will have a
      single random value embedded in the trace. This is likely not the
      intended behavior. One workaround is to wrap ``oneflow.randn`` in a ``oneflow.fx.wrap`` function and call that instead.

    ::

        @oneflow.fx.wrap
        def oneflow_randn(x, shape):
            return oneflow.randn(shape)

        def f(x):
            return x + oneflow_randn(x, 5)
        fx.symbolic_trace(f)

   -  This behavior may be fixed in a future release.

-  Type annotations

   -  Python 3-style type annotations (e.g.
      ``func(x : oneflow.Tensor, y : int) -> oneflow.Tensor``) are supported
      and will be preserved by symbolic tracing.
   -  Python 2-style comment type annotations
      ``# type: (oneflow.Tensor, int) -> oneflow.Tensor`` are not currently
      supported.
   -  Annotations on local names within a function are not currently
      supported.


API Reference
-------------

.. autofunction:: oneflow.fx.symbolic_trace

.. autofunction:: oneflow.fx.wrap

.. autoclass:: oneflow.fx.GraphModule
  :members:

  .. automethod:: __init__

.. autoclass:: oneflow.fx.Graph
  :members:

  .. automethod:: __init__

.. autoclass:: oneflow.fx.Node
  :members:

.. autoclass:: oneflow.fx.Tracer
  :members:
  :inherited-members:

.. autoclass:: oneflow.fx.Proxy

.. autoclass:: oneflow.fx.Interpreter
  :members:

.. autoclass:: oneflow.fx.Transformer
  :members:

.. autofunction:: oneflow.fx.replace_pattern

