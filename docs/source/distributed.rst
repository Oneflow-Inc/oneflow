oneflow.distributed
=========================================================

.. note ::
    Please refer to `OneFlow Distributed Overview <https://docs.oneflow.org/master/parallelism/01_introduction.html>`__
    for a brief introduction to all features related to distributed training.

OneFlow provides two ways to accomplish `Distributed Training`:

- The first way is that users are recommended to use OneFlow's global Tensor for distributed training. Global Tensor regards the computing cluster as a supercomputing device, allowing users to write distributed training code just like in a single-machine environment.

- OneFlow also provides a DDP（DistributedDataParallel） module aligned with PyTorch. DDP has been well-known and widely used in data parallelism by the majority of PyTorch users. Also see `PyTorch DDP introduction <https://pytorch.org/docs/1.10/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_.



Basic
-------------------------------
When you start distributed training in OneFlow, the following functions can be used.

.. currentmodule:: oneflow.env

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_world_size
    get_rank
    get_local_rank
    get_node_size
    init_rdma
    rdma_is_initialized


`Global Tensor`
--------------------------------------------------------------

Construct `Global Tensor`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A `Global Tensor` can be created with a ``placement`` and a ``sbp``. The ``placement`` describes the physical devices of the global tensor will be allocated, and the ``sbp`` describes its distribution among these devices.

::

    >>>import oneflow as flow
    >>> # Place a global tensor on cuda device of rank(process) 0 and 1
    >>> placement = flow.placement(type="cuda", ranks=[0, 1])
    >>> # Each rank's local data is a part data as a result of spliting global data on dim 0
    >>> sbp = flow.sbp.split(dim=0)
    >>> # Create a global tensor by randn
    >>> x = flow.randn(4, 5, placement=placement, sbp=sbp)
    >>> x.shape
    oneflow.Size([4, 5])


Convert `Local Tensor` to `Global Tensor`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``Tensor.to_global`` interface, `Local Tensor` can create a `Global Tensor` and use that `Local Tensor` as its local component at the current node.

Two `local tensors` with the shape of ``(2,5)`` are created separately on two devices. While after the ``to_global`` method, the `global tensor` with a shape of ``(4,5)`` is obtained.

Code running on Node 0

::

    import oneflow as flow

    x = flow.randn(2,5)
    placement = flow.placement("cuda", [0,1])
    sbp = flow.sbp.split(0)
    x_global = x.to_global(placement=placement, sbp=sbp)
    x_global.shape

Code running on Node 1

::

    import oneflow as flow

    x = flow.randn(2,5)
    placement = flow.placement("cuda", [0,1])
    sbp = flow.sbp.split(0)
    x_global = x.to_global(placement=placement, sbp=sbp)
    x_global.shape

Redistribute `Global Tensor`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Redistributing a `Global Tensor` means moving its data to another device group (or placement), or changing its data distribution (or SBP) across the group, or both at the same time. The redistributed tensor is still a `Global Tensor`.

::

    >>> import oneflow as flow
    >>> x = flow.tensor([1.0, 2.0], placement=flow.placement("cuda", ranks=[0, 1]), sbp=flow.sbp.split(0))
    >>> y = x.to_global(placement=flow.placement("cuda", ranks=[2, 3]), sbp=flow.sbp.broadcast)

According to the operator's semantics, OneFlow defines a sequence of valid input and output SBP combinations for each built-in operator. So OneFlow could automatically redistribute the `Global Tensor` to satisfy the operator's SBP requirements for its input Tensor. For example, the following code:

::

    >>> import oneflow as flow
    >>> x = flow.randn(4, 4, 
            placement=flow.placement("cuda", ranks=[0, 1]), 
            sbp=flow.sbp.split(0))
    >>> y = flow.randn(4, 4, 
            placement=flow.placement("cuda", ranks=[0, 1]), 
            sbp=flow.sbp.split(1))
    >>> z = x + y

When ``x + y`` is executed, since x is split along dimension ``0`` and y is split along dimension ``1``, their local components at each node can not be added directly, then OneFlow will automatically redistribute one of x and y to make them have the same SBP, and complete the add operation successfully.

.. note ::
    - Global Tensor can not be used in combination with DDP currently.
    - Global Tensor requires all devices to execute at the same pace, otherwise, it may cause multi-process deadlock.

Get Local Tensor from Global Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With ``Tensor.to_local`` interface, the `Global Tensor` can return its local component at the current node.

::

    y = x.to_local()
    y.is_local
    True
    y
    tensor([[ 2.9186e-01, -3.9442e-01,  4.7072e-04, -3.2216e-01,  1.7788e-01],
                [-4.5284e-01,  1.2361e-01, -3.5962e-01,  2.6651e-01,  1.2951e+00]],
            device='cuda:0', dtype=oneflow.float32)


DistributedDataParallel
--------------------------------------------------------------

For more information about DistributedDataParallel, see ``nn.parallel.DistributedDataParallel``

The following script shows the process of using ``oneflow.nn.parallel.DistributedDataParallel`` for training data parallel: 

.. code-block:: 

    import oneflow as flow
    from oneflow.nn.parallel import DistributedDataParallel as ddp

    train_x = [
        flow.tensor([[1, 2], [2, 3]], dtype=flow.float32),
        flow.tensor([[4, 6], [3, 1]], dtype=flow.float32),
    ]
    train_y = [
        flow.tensor([[8], [13]], dtype=flow.float32),
        flow.tensor([[26], [9]], dtype=flow.float32),
    ]


    class Model(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.lr = 0.01
            self.iter_count = 500
            self.w = flow.nn.Parameter(flow.tensor([[0], [0]], dtype=flow.float32))

        def forward(self, x):
            x = flow.matmul(x, self.w)
            return x


    m = Model().to("cuda")
    m = ddp(m)
    loss = flow.nn.MSELoss(reduction="sum")
    optimizer = flow.optim.SGD(m.parameters(), m.lr)

    for i in range(0, m.iter_count):
        rank = flow.env.get_rank()
        x = train_x[rank].to("cuda")
        y = train_y[rank].to("cuda")

        y_pred = m(x)
        l = loss(y_pred, y)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{m.iter_count} loss:{l}")

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print(f"\nw:{m.w}")

There are only two differences between the data parallelism training code and the stand-alone single-card script:

- Use `DistributedDataParallel` to wrap the module object (`m = ddp(m)`)
- Use `get_rank` to get the current device number and distribute the data to the device.

Then use `launcher` to run the script, leave everything else to OneFlow, which makes distributed training as simple as stand-alone single-card training:

::

    python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py


Communication collectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: oneflow.comm
    
.. autosummary::
    :toctree: generated
    :nosignatures:

        all_reduce
        all_gather
        all_gather_into_tensor
        all_to_all
        broadcast
        barrier
        gather
        reduce
        reduce_scatter
        reduce_scatter_tensor
        recv
        scatter
        send

We also provide PyTorch-compatible APIs for communication collectives, for example, `oneflow.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)`. For more information, see `PyTorch Distributed Communication <https://pytorch.org/docs/stable/distributed.html>`_. Note that we currently only support op=ReduceOp.SUM, group=None and async_op=False in these operations.

Launching distributed training
--------------------------------------------------------------

.. currentmodule:: oneflow.distributed

run commands below to see more about usage.

::

    python3 -m oneflow.distributed.launch -h

.. code-block::

    usage: launch.py [-h] [--nnodes NNODES] [--node_rank NODE_RANK]
                 [--nproc_per_node NPROC_PER_NODE] [--master_addr MASTER_ADDR]
                 [--master_port MASTER_PORT] [-m] [--no_python]
                 [--redirect_stdout_and_stderr] [--logdir LOGDIR]
                 training_script ...

    OneFlow distributed training launch helper utility that will spawn up multiple
    distributed processes

    positional arguments:
    training_script       The full path to the single GPU training program/script to be
                            launched in parallel, followed by all the arguments for the
                            training script
    training_script_args

    optional arguments:
    -h, --help            show this help message and exit
    --nnodes NNODES       The number of nodes to use for distributed training
    --node_rank NODE_RANK
                            The rank of the node for multi-node distributed training
    --nproc_per_node NPROC_PER_NODE
                            The number of processes to launch on each node, for GPU
                            training, this is recommended to be set to the number of GPUs in
                            your system so that each process can be bound to a single GPU.
    --master_addr MASTER_ADDR
                            Master node (rank 0)'s address, should be either the IP address
                            or the hostname of node 0, for single node multi-proc training,
                            the --master_addr can simply be 127.0.0.1
    --master_port MASTER_PORT
                            Master node (rank 0)'s free port that needs to be used for
                            communication during distributed training
    -m, --module          Changes each process to interpret the launch script as a python
                            module, executing with the same behavior as'python -m'.
    --no_python           Do not prepend the training script with "python" - just exec it
                            directly. Useful when the script is not a Python script.
    --redirect_stdout_and_stderr
                            write the stdout and stderr to files 'stdout' and 'stderr'. Only
                            available when logdir is set
    --logdir LOGDIR       Relative path to write subprocess logs to. Passing in a relative
                            path will create a directory if needed. Note that successive
                            runs with the same path to write logs to will overwrite existing
                            logs, so be sure to save logs as needed.
