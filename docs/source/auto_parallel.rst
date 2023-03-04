Auto Parallelism
====================================================

As the scale of deep-learning models grows larger and larger, distributed training,
or parallelism, is needed. Data parallelism and model parallelism has been designed
to speed up the training and solve memory issues.

In oneflow, SBP signature enables users to configure parallelism policy easily.
However, users still need to specify the SBP property for each operator, or most of them.
Users might spend a couple of days digging into the detail of parallelism and get a
low throughput just because of a slight mistake in the configuration of SBP signature.

.. note::

   It only works on :doc:`graph` mode.


Our strength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get rid of all those configurations for SBP signatures, we developed auto parallelism.
Still, configurations of placement are necessary and we have not supported auto placement
yet. If you read this paragraph before you rush into any SBP stuff, then congratulation,
you do not need to learn SBPs. You can start writing your code as you did under CPU mode.
Our auto parallelism would generate a fast strategy customized for your specific models,
the size of parameters, and the number of available GPUs.


How to use auto parallelism?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You just need to simply enable the configuration settings in the model
of :doc:`graph` .

Example::

    import oneflow as flow
    class SubclassGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__() # MUST be called
            # auto parallelism configuration
            self.config.enable_auto_parallel(True)
            # other configurations about auto parallelism
            # ......

        def build(self):
            pass

.. warning::

   If you enable auto parallelism, OneFlow will take care of the SBP configurations
   of operators except for explicit ``to_global`` functions.


Configuration API for auto parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: oneflow.nn.graph.graph_config.GraphConfig

.. autosummary::
    :toctree: generated
    :nosignatures:

    enable_auto_parallel
    enable_auto_parallel_ignore_user_sbp_config
    set_auto_parallel_computation_cost_ratio
    set_auto_parallel_wait_time
    enable_auto_parallel_trunk_algo
    enable_auto_parallel_sbp_collector
    enable_auto_memory

