oneflow.distributed
=========================================================

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