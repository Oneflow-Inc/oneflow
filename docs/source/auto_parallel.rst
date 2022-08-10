Auto Parallelism
====================================================

TODO: A brief introduction

.. note::

   It only works on :doc:`graph` mode.


Why do you need auto parallelism?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Advantages of using automatic parallelism.


What can automatic parallelism do for you?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Auto parallelism can do what for users.


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

   If you enable auto parallelism, all ``oneflow.sbp`` operations will be ignored.


Configuration API for auto parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: oneflow.nn.graph.graph_config.GraphConfig

.. autosummary::
    :toctree: generated
    :nosignatures:

    enable_auto_parallel
    enable_auto_parallel_prune_parallel_cast_ops
    set_auto_parallel_computation_cost_ratio
    set_auto_parallel_wait_time
    enable_auto_parallel_mainstem_algo
    enable_auto_parallel_sbp_collector

