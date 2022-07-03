oneflow.nn.Graph
============================================================
Currently, there are two main ways to run models in deep learning frameworks, namely dynamic graphs and static graphs , which are also conventionally referred to as :ref:`dynamic graph` and :ref:`static graph` in OneFlow.

Both approaches have their advantages and disadvantages, and OneFlow provides support for both approaches, with Eager mode being the default.

Generally speaking, dynamic graphs are easier to use and static graphs have more performance advantages. :class:`oneflow.nn.Graph` module is provided by OneFlow to allow users to build static graphs and train models with Eager-like programming conventions

.. contents:: Graph
    :depth: 2
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :backlinks: top

.. _dynamic graph:

Eager Mode
------------------------------------------------------------

OneFlow runs in Eager mode by default.

The following script, using the CIFAR10 dataset, trains the mobilenet_v2 model:

.. code-block:: 

        import oneflow as flow
        import oneflow.nn as nn
        import flowvision
        import flowvision.transforms as transforms

        BATCH_SIZE=64
        EPOCH_NUM = 1

        DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
        print("Using {} device".format(DEVICE))

        training_data = flowvision.datasets.CIFAR10(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
        source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz",
        )

        train_dataloader = flow.utils.data.DataLoader(
        training_data, BATCH_SIZE, shuffle=True
        )

        model = flowvision.models.mobilenet_v2().to(DEVICE)
        model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
        model.train()

        loss_fn = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


        for t in range(EPOCH_NUM):
        print(f"Epoch {t+1}\n-------------------------------")
        size = len(train_dataloader.dataset)
        for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * BATCH_SIZE
        if batch % 5 == 0:
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

.. _static graph:

Static Graph Mode
------------------------------------------------------------


Constructing it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: oneflow.nn.Graph

.. autosummary::
    :toctree: generated
    :nosignatures:

    __init__
    build
    __call__



Graph Config option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    

Block Config option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: oneflow.nn.graph.block_config.BlockConfig

.. autosummary::
    :toctree: generated
    :nosignatures:

    stage_id
    set_stage
    activation_checkpointing

Base class
-------------------
.. currentmodule:: oneflow.nn.Graph

.. autosummary::
    :toctree: generated
    :nosignatures:

    add_optimizer
    set_grad_scaler
    state_dict
    load_state_dict
    name
    debug
    __repr__





