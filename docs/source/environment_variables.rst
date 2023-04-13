Environment Variables
================================================

OneFlow has an extensive set of environment variables to tune for specific usage.

`ONEFLOW_COMM_NET_IB_HCA <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/comm_network/ibverbs/ibverbs_comm_network.cpp#L47>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

When there are multiple IB NIC(which can be checked by ``ibstatus`` on the server, the system uses the first IB NIC for comm_net communication by default.

When this environment variable is set, the system will check all IB NIC and find the NIC with the corresponding name. `#5626 <https://github.com/Oneflow-Inc/oneflow/pull/5626>`_

Values accepted
^^^^^^^^^^^^^^^
The default value is empty, such as ``mlx5_0:1``、 ``mlx5_1:1``. When the port is 0, the default value is 1, representing the first port.

`ONEFLOW_COMM_NET_IB_GID_INDEX <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/comm_network/ibverbs/ibverbs_comm_network.cpp#L142>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For the query of `ibv_query_gid <https://www.ibm.com/docs/en/aix/7.2?topic=management-ibv-query-gid>`_, and 0 represents success. It often used with ``ONEFLOW_COMM_NET_IB_HCA``. GID means the Global ID, QP under RoCE network must be built by this value, instead of just using the LID as in the IB network. `#5626 <https://github.com/Oneflow-Inc/oneflow/pull/5626>`_

Values accepted
^^^^^^^^^^^^^^^
The default value is 0, representing the port index value

`ONEFLOW_COMM_NET_IB_QUEUE_DEPTH <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/comm_network/ibverbs/ibverbs_qp.cpp#L44>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Queue length of jobs in IB network.

This value effectively controls the size of the module without instead of using IB's default size, such as ``ONEFLOW_COMM_NET_IB_MEM_BLOCK_SIZE``.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``1024``, receiving ``int64_t``. The system would compare with ``max_qp_wr`` (Maximum number of outstanding WR on any work queue), and take the smaller one.

`ONEFLOW_COMM_NET_IB_MEM_BLOCK_SIZE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/comm_network/ibverbs/ibverbs_qp.cpp#L68>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The size of the module read when communicating.

The value can calculate the amount of module, and transmit it after encapsulation.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``8388608`` (8M)

`ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/ep/cuda/cuda_device.cpp#L59>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Represents stream, and marks Blocking synchronization in cuda. `Detailed information <https://www.cnblogs.com/1024incn/p/5891051.html>`_, `#5612 <https://github.com/Oneflow-Inc/oneflow/pull/5612>`_, `#5837 <https://github.com/Oneflow-Inc/oneflow/pull/5837>`_

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_LIBIBVERBS_PATH <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/platform/lib/ibv_wrapper.cpp#L24>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To load the DynamicLibrary by dlopen at runtime, to find symbols of ibverbs functions by dlopen without linking during compile for better compatibility. `#4852 <https://github.com/Oneflow-Inc/oneflow/pull/4852>`_.

If it failed, it will output ``libibverbs not available, ibv_fork_init skipped``, if it worked, the ``import oneflow`` will output such as ``loaded library: /usr/lib/x86_64-linux-gnu/libibverbs.so.1``

Values accepted
^^^^^^^^^^^^^^^
The default value is empty, but will load ``libibverbs.so.1``, ``libibverbs.so``.

`ONEFLOW_DEBUG_MODE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/common/env_var/debug_mode.h#L23>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Enable ``debug`` mode, ``ONEFLOW_DEBUG`` can do.

If ``debug`` mode is on, it will output more INFO level logs, different ``prototxt`` and ``dot`` to files. The automatically inserted boxing information will be printed to the log file under eager global mode.

Values accepted
^^^^^^^^^^^^^^^
The default value is empty, but will receive any string.

`ONEFLOW_DRY_RUN <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job/resource_desc.cpp#L65>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Only for test running, it can generate log files like ``dot``.

Exit once the test is succeed, do not try real training.

Values accepted
^^^^^^^^^^^^^^^
The default value is empty, but will receive any string.

`ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/lazy/stream_context/cuda/cuda_stream_context.cpp#L66>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Only used when debugging because the performance would be affected, it could detect which op in the network appears nan or inf.

It will create ``CpuCheckNumericsKernelObserver`` under ``cpu`` , and ``CudaCheckNumericsKernelObserver`` under ``cuda`` `#6052 <https://github.com/Oneflow-Inc/oneflow/pull/6052>`_ .

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_DEBUG_KERNEL_SYNC_CHECK <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job/env_global_objects_scope.cpp#L193>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Only used when debugging because the performance would be affected.

It will create ``SyncCheckKernelObserver`` and will be synced after each kernel.

It could be used to debug cuda errors. `#6052 <https://github.com/Oneflow-Inc/oneflow/pull/6052>`_

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_PROFILER_KERNEL_PROFILE_CUDA_MEMORY_BANDWIDTH <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/profiler/kernel.cpp#L34>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Used when generate profiler files by nsys.

Profiler is only valid for lazy temporarily.

It can estimate the memory bandwidth reached by kernel by counting the execution time of the GPU kernel and the size of the input and output memory, and help find potential kernels that can be optimized. `Details <https://github.com/Oneflow-Inc/oneflow/blob/02e29f9648f63a4d936cd818061e90064d027005/oneflow/core/profiler/kernel.cpp#L53>`_

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``. When using, the compiled package needs to enable ``BUILD_PROFILER``.

`ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/profiler/kernel.cpp#L36>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The same as above. collect `op name <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/profiler/kernel.cpp#L62>`_

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``. When using, the compiled package needs to enable ``BUILD_PROFILER``.

`ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job/env_global_objects_scope.cpp#L199>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Only use blob_access_checker after enabling, because blob_access_checker is for correctness assurance, and closing it in some cases can increase the kernel overhead. `#5728 <https://github.com/Oneflow-Inc/oneflow/pull/5728>`_

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/kernel/user_kernel.cpp#L692>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Takes effect under ``WITH_CUDA_GRAPHS`` and the default value is ``false``. It uses more memory, so when there's just enough memory, it won't run.

Turning on CUDA_GRAPH will use up more memory CUDA Graphs support. `#5868 <https://github.com/Oneflow-Inc/oneflow/pull/5868>`_

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/thread/thread.cpp#L30>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

LightActor is a new type of Actor that only handles NormalForward and similar tasks where all regst_num is 1 or tasks with only one kernel. `#5868 <https://github.com/Oneflow-Inc/oneflow/pull/5868>`_. ``export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1`` (Would use more memories), ``export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1``, ``export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1``, ``export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1``, ``export ONEFLOW_STREAM_REUSE_CUDA_EVENT=1`` can be used together.

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/thread/thread.cpp#L29>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

`#5720 <https://github.com/Oneflow-Inc/oneflow/pull/5720>`_. It is used to enable local message queue, ``oneflow.config.thread_enable_local_message_queue(True)`` is no longer used.

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``false``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_PERSISTENT_IN_STREAM_BUFFER_SIZE_BYTES <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/persistence/persistent_in_stream.cpp#L30>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Represents the size of each read from disk. `#5162 <https://github.com/Oneflow-Inc/oneflow/pull/5162>`_

Values accepted
^^^^^^^^^^^^^^^
The default value is empty. If an invalid string or negative number is entered, the default value would be ``32 * 1024``; 32KB.

`ONEFLOW_DECODER_ENABLE_NVJPEG_HARDWARE_ACCELERATION <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/kernel/image_decoder_random_crop_resize_kernel.cpp#L290>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

``NVJPEG_VER_MAJOR`` need to be bigger than ``11``. It can accelerate nvjpeg hardware, warm up jpeg decoder and hw_jpeg decoder, `#5851 <https://github.com/Oneflow-Inc/oneflow/pull/5851>`_.

Hardware JPEG decoder and NVIDIA nvJPEG library on NVIDIA A100 GPUs

Values accepted
^^^^^^^^^^^^^^^
Define and set to ``true``, and would be ``true`` only when the value is ``1``, ``true``, ``yes``, ``on`` and ``y``.

`ONEFLOW_SERVING_DEBUG <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/api/cpp/framework/graph.cpp#L213>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For printing information of OneFlow Serving Debug

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_DISABLE_VIEW <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/framework/tensor_methods.cpp#L35>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To disable view mechanism, which means op related to view would stop running.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/auto_parallel/boxing_collector.cpp#L82>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to disable Middle Node. When it is false, all inter-SBP communication is supported

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_ONE_EMBEDDING_DISABLE_NUMA_AWARE_ALLOCATION <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/embedding/full_cache.cu#L414>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to disable NUMA_AWARE memory allocation when the OneEmbedding module allocates video memory.

NUMA_AWARE memory allocation means that when allocating pinned host memory, the cpu close to the gpu will be considered (for example, if it is gpu 0 1, memory will be allocated on cpu0)

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_EP_CUDA_ENABLE_TF32_EXECUTION <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/ep/cuda/cuda_stream.cpp#L96>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to allow CUDA to use TF32 numeric types for computation

Values accepted
^^^^^^^^^^^^^^^
The default value is ``true``

`ONEFLOW_FUNCTOR_DISABLE_FUSED_MLP <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/functional/impl/nn_functor.cpp#L554>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to disable the fused_mlp operator implemented by cublasLt in FusedMLPFunctor, if disabled, it will degenerate into a multiple matrix multiplication operation.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_INDEPENTENT_STREAM <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job_rewriter/replace_embedding_ops_pass.cpp#L192>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to put the EmbeddingShuffle of the OneEmbedding module on a separate stream for overlapping execution.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_ONE_EMBEDDING_GRADIENT_SHUFFLE_USE_FP16 <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job_rewriter/replace_embedding_ops_pass.cpp#L209>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to allow the EmbeddingGradientShuffle operator of the OneEmbedding module to use the FP16 data type in the AMP case.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``true``

`ONEFLOW_ONE_EMBEDDING_NOT_FUSE_CAST_TO_UPDATE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job_rewriter/replace_embedding_ops_pass.cpp#L260>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to disable the fusion of cast type conversion and parameter update of OneEmbedding parameters into one operator in the case of AMP

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS_DUMP <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/kernel/cpu_numerics_kernel_observer.cpp#L65>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

When the value appears NaN or Inf, save the data Dump.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_MLIR_ENABLE_IR_PRINTING <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/ir/lib/OneFlow/Passes.cpp#L768>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control whether to print ir when running each pass when debugging

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_MLIR_STDOUT <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/ir/oneflow-extension/extension.cpp#L151>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control whether MLIR outputs log information in the console

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_MLIR_DUMP_IR <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/ir/oneflow-extension/extension.cpp#L152>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control whether to dump ir files

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_MLIR_ENABLE_ROUND_TRIP <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/ir/oneflow-extension/ir_pass.cpp#L157>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control whether Oneflow Job goes into MLIR

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_KERNEL_REDUCE_SUM_USE_MATMUL <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/user/kernels/reduce_kernel.cpp#L333>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

whether to use matrix multiplication for reduce_sum

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM <https://github.com/Oneflow-Inc/oneflow/blob/dd580f21ffb6e4d23a899c7e0ac6d2bc502f3f1a/oneflow/core/job_rewriter/fuse_embedding_interaction_pass.cpp#L35>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Whether to quantify the shuffle application communication in the case of OneEmbedding multi-card

Values accepted
^^^^^^^^^^^^^^^
The default value is ``false``

`ONEFLOW_TENSOR_BUFFER_ALIGNED_SIZE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/common/tensor_buffer.cpp#L29>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Align size when allocating TensorBuffer memory

Values accepted
^^^^^^^^^^^^^^^
The default value is ``1024``

`ONEFLOW_TENSOR_BUFFER_POOL_THREAD_LOCAL_CACHE_SIZE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/common/tensor_buffer.cpp#L206>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control the size of ``thread_local_cache`` in TensorBufferPool

Values accepted
^^^^^^^^^^^^^^^
The default value is ``64``

`ONEFLOW_GRPC_MAX_MESSAGE_BYTE_SIZE <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/control/ctrl_service.cpp#L45>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Set the maximum size of the gRPC transport message

Values accepted
^^^^^^^^^^^^^^^
The default value is ``-1``

`ONEFLOW_ONE_EMBEDDING_PERSISTENT_TABLE_CAPACITY_HINT <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/embedding/persistent_table.cpp#L410>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control the initial capacity of the PersistentTable of OneEmbedding to avoid frequent expansion

Values accepted
^^^^^^^^^^^^^^^
OneEmbedding will calculate according to the actual situation, and users can also choose to configure a larger capacity.

`ONEFLOW_ONE_EMBEDDING_PERSISTENT_TABLE_NUM_WORKERS <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/embedding/persistent_table.cpp#L435>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The number of threads used for reading and writing the PersistentTable of OneEmbedding

Values accepted
^^^^^^^^^^^^^^^
The default value is ``4``

`ONEFLOW_EP_CUDA_CONST_BUFFER_ELEMENT_COUNT <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/ep/cuda/cuda_device.cpp#L62>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Specify the size of the all zero and all one buffers on the CUDA device.

This buffer can be used with matrix multiplication to implement operations such as reduce_sum

Values accepted
^^^^^^^^^^^^^^^
The default value is ``1024x1024``

`OMP_NUM_THREADS <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job/env_global_objects_scope.cpp#L96>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Set the number of threads used by OMP

Values accepted
^^^^^^^^^^^^^^^
The default value will be generated by specific `computational logic <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job/env_global_objects_scope.cpp#L106-L108>`_.

`SBP_INFER_RULE_TAG <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/operator/operator.cpp#L718>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Specify SBP derivation rules

Values accepted
^^^^^^^^^^^^^^^
When the default vaule is ``1`` , select the SBP that satisfies the producer or the SBP with the smallest cost as much as possible.

When the default value is ``2``, select the SBP that matches the most.

When the default value is ``3``, select the SBP with the smallest cost.

`ONEFLOW_TENSOR_BUFFER_GROWTH_FACTOR <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/common/tensor_buffer.cpp#L35>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control the growth factor of TensorBuffer

Values accepted
^^^^^^^^^^^^^^^
The default value is ``1.0``

`ONEFLOW_TENSOR_BUFFER_SHRINK_FACTOR <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/common/tensor_buffer.cpp#L41>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Controls the shrink factor of TensorBuffer

Values accepted
^^^^^^^^^^^^^^^
The default value is ``0.7``

`ONEFLOW_TENSOR_BUFFER_POOL_SIZE_FACTOR <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/common/tensor_buffer.cpp#L200>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Controls the size factor of TensorBuffer

Values accepted
^^^^^^^^^^^^^^^
The default value is ``2.0``

`AUTO_PARALLEL_TRANSFER_COST <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/framework/sbp_infer_util.cpp#L544>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Control the size of the automatic parallel transfer cost

Values accepted
^^^^^^^^^^^^^^^
The default value is ``1.65e8``


`ONEFLOW_DEBUG_PASS <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/job/job_build_and_infer_ctx.cpp#L991>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Pass names and print job before and after a specific pass, such as ``export ONEFLOW_DEBUG_PASS="FuseAddToOutputPass``.

Or ALL, print job before and after a specific pass, such as ``export ONEFLOW_DEBUG_PASS="ALL"``.

Values accepted
^^^^^^^^^^^^^^^
The default value is ``empty``

`ONEFLOW_PROFILER_HOST_THREAD_NAME_PREFIX <https://github.com/Oneflow-Inc/oneflow/blob/v0.9.0/oneflow/core/profiler/profiler.cpp#L39>`_
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Add a prefix to the name of the named host thread in the profiling context to facilitate sorting in the visualization tool (nsight)

Values accepted
^^^^^^^^^^^^^^^
The default value is ``empty``
