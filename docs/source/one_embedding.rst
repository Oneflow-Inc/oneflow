oneflow.one_embedding
===================================

Embedding is an important component of recommender system, and it has also spread to many fields outside recommender systems. Each framework provides basic operators for Embedding, for example, ``flow.nn.Embedding`` in OneFlow:

::

    import numpy as np
    import oneflow as flow
    indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
    embedding = flow.nn.Embedding(10, 3)
    y = embedding(indices)


OneEmbedding is the large-scale Embedding solution that OneFlow provides to solve the problem of large-scale deep recommender systems. OneEmbedding has the following advantages compared to ordinary opeartors:

    - With Flexible hierarchical storage, OneEmbedding can place the Embedding table on GPU memory, CPU memory or SSD, and allow high-speed devices to be used as caches for low-speed devices to achieve both speed and capacity.

    - OneEmbedding supports dynamic expansion.

.. note ::
    Please refer to `Large-Scale Embedding Solution: OneEmbedding <https://docs.oneflow.org/en/master/cookies/one_embedding.html>`__
    for a brief introduction to all features related to OneEmbedding.

Configure Embedding Table 
----------------------------------

OneEmbedding supports simultaneous creation of multiple Embedding table. The following codes configured three Embedding tables.

.. code-block:: 

    import oneflow as flow
    import oneflow.nn as nn
    import numpy as np

    tables = [
        flow.one_embedding.make_table_options(
            flow.one_embedding.make_uniform_initializer(low=-0.1, high=0.1)
        ),
        flow.one_embedding.make_table_options(
            flow.one_embedding.make_uniform_initializer(low=-0.05, high=0.05)
        ),
        flow.one_embedding.make_table_options(
            flow.one_embedding.make_uniform_initializer(low=-0.15, high=0.15)
        ),
    ]

When configuring the Embedding table, you need to specify the initialization method. The above Embedding tables are initialized in the ``uniform`` method. The result of configuring the Embedding table is stored in the ``tables`` variable

.. autofunction:: oneflow.one_embedding.make_table_options
.. autofunction:: oneflow.one_embedding.make_table

initialization method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: oneflow.one_embedding

.. autosummary::
    :toctree: generated
    :nosignatures:

    make_uniform_initializer
    make_normal_initializer


Configure the Storage Attribute of the Embedding Table
--------------------------------------------------------------------
Then run the following codes to configure the storage attribute of the Embedding table:

.. code-block:: 

    store_options = flow.one_embedding.make_cached_ssd_store_options(
    cache_budget_mb=8142,
    persistent_path="/your_path_to_ssd", 
    capacity=40000000,
    size_factor=1,              
    physical_block_size=4096
    )

Storage Method
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: oneflow.one_embedding

.. autosummary::
    :toctree: generated
    :nosignatures:

    make_device_mem_store_options
    make_cached_ssd_store_options 
    make_cached_host_mem_store_options

.. note ::
    
    Please refer to `Large-Scale Embedding Solution: OneEmbedding <https://docs.oneflow.org/en/master/cookies/one_embedding.html#feature-id-and-dynamic-insertion>`__
    for a brief introduction to learn about How to Choose the Proper Storage Configuration


Instantiate Embedding
--------------------------------------------------------------------
After the above configuration is completed, you can use MultiTableEmbedding to get the instantiated Embedding layer.

.. code-block:: 

    embedding_size = 128
    embedding = flow.one_embedding.MultiTableEmbedding(
        name="my_embedding",
        embedding_dim=embedding_size,
        dtype=flow.float,
        key_type=flow.int64,
        tables=tables,
        store_options=store_options,
    )

    embedding.to("cuda")

.. note ::
    
    Please refer to `Large-Scale Embedding Solution: OneEmbedding <https://docs.oneflow.org/en/master/cookies/one_embedding.html#feature-id-and-multi-table-query>`__
    for a brief introduction to learn about Feature ID and Multi-Table Query.


MultiTableEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: oneflow.one_embedding.MultiTableEmbedding

.. currentmodule:: oneflow.one_embedding.MultiTableEmbedding

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    forward
    save_snapshot
    load_snapshot

MultiTableMultiColumnEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: oneflow.one_embedding.MultiTableMultiColumnEmbedding

.. currentmodule:: oneflow.one_embedding.MultiTableMultiColumnEmbedding

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    forward
    save_snapshot
    load_snapshot

Construct Graph for Training
--------------------------------------------------------------------
OneEmbedding is only supported in Graph mode.

.. code-block:: 

    num_tables = 3
    mlp = flow.nn.FusedMLP(
        in_features=embedding_size * num_tables,
        hidden_features=[512, 256, 128],
        out_features=1,
        skip_final_activation=True,
    )
    mlp.to("cuda")

    class TrainGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()
            self.embedding_lookup = embedding
            self.mlp = mlp
            self.add_optimizer(
                flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.0)
            )
            self.add_optimizer(
                flow.optim.SGD(self.mlp.parameters(), lr=0.1, momentum=0.0)
            )
        def build(self, ids):
            embedding = self.embedding_lookup(ids)
            loss = self.mlp(flow.reshape(embedding, (-1, num_tables * embedding_size)))
            loss = loss.sum()
            loss.backward()
            return loss

.. note ::
    
    Please refer to `Distributed Training: OneEmbedding <https://docs.oneflow.org/en/master/parallelism/01_introduction.html>`__
    for a brief introduction to learn about Graph For Training


Persistent Read & Write
-----------------------------------------------
.. currentmodule:: oneflow.one_embedding

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    make_persistent_table_reader
    make_persistent_table_writer

.. automodule:: oneflow.one_embedding
    :members: Ftrl

