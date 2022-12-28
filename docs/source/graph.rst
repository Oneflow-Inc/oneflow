oneflow.nn.Graph
============================================================
Base class for running neural networks in Static Graph Mode.

Currently, there are two main ways to run models in deep learning frameworks, namely dynamic graphs and static graphs , which are also conventionally referred to as :ref:`dynamic graph` and :ref:`static graph` in OneFlow.

Both approaches have their advantages and disadvantages, and OneFlow provides support for both approaches, with Eager mode being the default.

Generally speaking, dynamic graphs are easier to use and static graphs have more performance advantages. :class:`oneflow.nn.Graph` module is provided by OneFlow to allow users to build static graphs and train models with Eager-like programming conventions.

.. contents:: oneflow.nn.Graph
    :depth: 2
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :backlinks: top

.. _dynamic graph:

Eager Mode to Static Graph Mode
------------------------------------------------------------

OneFlow runs in Eager mode by default.

OneFlow's nn.Graph is programmed in a style very similar to Eager Mode, so it is possible to make small changes and get large performance gains.

The following script shows the process of building a neural network in eager mode using the interface under ``oneflow.nn`` :


.. code-block:: 

    import oneflow as flow
    import oneflow.nn as nn

    class ModuleMyLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(flow.randn(in_features, out_features))
            self.bias = nn.Parameter(flow.randn(out_features))

        def forward(self, input):
            return flow.matmul(input, self.weight) + self.bias

    linear_model = ModuleMyLinear(4, 3)


Eager ``nn.Module`` can be reused by ``nn.Graph``. The above script for eager mode can be changed to static Graph mode by adding just a few lines of code, which consists of the following steps:

- Define your customized graph as a subclass of ``nn.Graph``
- At the beginning of __init__. Call super().__init__() to let OneFlow do the necessary initialization of the Graph
- Reuse the ``nn.Module`` object in Eager mode in __init__ (self.model = model)
- Describe the computation in the ``build`` method
- Instantiate your graph then call it.

.. code-block:: 

    class GraphMyLinear(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = linear_model

        def build(self, input):
            return self.model(input)

    graph_mylinear = GraphMyLinear()
    input = flow.randn(1, 4)
    out = graph_mylinear(input)
    print(out)

    tensor([[-0.3298, -3.7907,  0.1661]], dtype=oneflow.float32)

.. _static graph:

Static Graph Mode
------------------------------------------------------------


Constructing a Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Base class for training or evaluating a neural network in static graph mode.

.. currentmodule:: oneflow.nn.Graph

.. autosummary::
    :toctree: generated
    :nosignatures:

    __init__
    build
    add_optimizer
    set_grad_scaler

Executing a Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Call a nn.Graph instance to run a customized graph.

.. currentmodule:: oneflow.nn.Graph

.. autosummary::
    :toctree: generated
    :nosignatures:

    __call__



Config options on a Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Optimization options of a nn.Graph.

.. currentmodule:: oneflow.nn.graph.graph_config.GraphConfig

.. autosummary::
    :toctree: generated
    :nosignatures:

    enable_amp
    enable_zero
    allow_fuse_model_update_ops
    allow_fuse_add_to_output
    allow_fuse_cast_scale
    set_gradient_accumulation_steps
    enable_cudnn_conv_heuristic_search_algo
    enable_straighten_algorithm
    enable_compress_memory
    

Config options on a GraphModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GraphModule is the graph representation of a nn.Module in a nn.Graph.

When an nn.Module is added into an nn.Graph, it is wrapped into a ProxyModule. The ProxyModule has a GraphModule inside it.
You can get and set the GraphModule to enable graph optimization on the nn.Module.

.. currentmodule:: oneflow.nn.graph.graph_block.GraphModule

.. autosummary::
    :toctree: generated
    :nosignatures:

    set_stage
    activation_checkpointing

Save & Load a Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: oneflow.nn.Graph

.. autosummary::
    :toctree: generated
    :nosignatures:

    state_dict
    load_state_dict


Debug a Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    __repr__
    debug
    name



