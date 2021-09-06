.. currentmodule:: oneflow.fx

oneflow.fx
=============

Overview
--------
**This feature is under a Beta release and its API may change.**

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
    import oneflow.nn as nn

    def transform(m: nn.Module,
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
FX transform, or you can run it. Ensuring that the inputs and outputs of your FX 
transform are a :class:`oneflow.nn.Module` will allow for composability.

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
    | call_function | add_1         | <built-in function add>    | (x, linear_weight) | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_module   | linear_1      | linear                     | (add_1,)           | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | relu_1        | relu                       | (linear_1,)        | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | sum_1         | sum                        | (relu_1,)          | {'dim': -1} |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | call_method   | topk_1        | topk                       | (sum_1, 3)         | {}          |
    +---------------+---------------+----------------------------+--------------------+-------------+
    | output        | output        | output                     | (topk_1,)          | {}          |
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
tracing and modify it. For example, let’s say we desire to replace
:func:`oneflow.add` calls with :func:`oneflow.mul` calls.


::

    import oneflow as flow

    # Sample module
    class M(flow.nn.Module):
        def forward(self, x, y):
            return flow.add(x, y)

    def transform(m: flow.nn.Module,
                    tracer_class : type = flow.fx.Tracer) -> flow.nn.Module:
        graph : flow.fx.Graph = tracer_class().trace(m)
        # FX represents its Graph as an ordered list of
        # nodes, so we can iterate through them.
        for node in graph.nodes:
            # Checks if we're calling a 
            #  (i.e:
            # flow.add)
            if node.op == 'call_method':
                # The target attribute is the method
                # that call_method calls.
                if node.target == flow.add:
                    node.target = flow.mul

        graph.lint() # Does some checks to make sure the
                        # Graph is well-formed.

        return flow.fx.GraphModule(m, graph)

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
        # Insert a new `call_method` node calling `flow.relu`
        new_node = traced.graph.call_method(
            flow.relu, args=(node,))

        # We want all places that used the value of `node` to
        # now use that value after the `relu` call we've added.
        # We use the `replace_all_uses_with` API to do this.
        node.replace_all_uses_with(new_node)

For simple transformations that only consist of substitutions, you can also
make use of the `subgraph rewriter. <https://github.com/pytorch/pytorch/blob/master/torch/fx/subgraph_rewriter.py>`__

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
   op <https://github.com/pytorch/examples/blob/master/fx/replace_op.py>`__
-  `Conv/Batch Norm
   fusion <https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50>`__
-  `replace_pattern: Basic usage <https://github.com/pytorch/examples/blob/master/fx/subgraph_rewriter_basic_use.py>`__
-  `Quantization <https://pytorch.org/docs/master/quantization.html#prototype-fx-graph-mode-quantization>`__
-  `Invert Transformation <https://github.com/pytorch/examples/blob/master/fx/invert.py>`__



