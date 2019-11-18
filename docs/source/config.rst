oneflow.config
===================================
.. currentmodule:: oneflow.config
.. automodule:: oneflow.config
    :members:
    :undoc-members:

.. autofunction:: gpu_device_num
.. autofunction:: cpu_device_num
.. autofunction:: machine
.. autofunction:: ctrl_port
.. autofunction:: data_port
.. autofunction:: comm_net_worker_num
.. autofunction:: max_mdsave_worker_num
.. autofunction:: compute_thread_pool_size
.. autofunction:: rdma_mem_block_mbyte
.. autofunction:: rdma_recv_msg_buf_mbyte
.. autofunction:: reserved_host_mem_mbyte
.. autofunction:: reserved_device_mem_mbyte
.. autofunction:: use_rdma
.. autofunction:: save_downloaded_file_to_local_fs
.. autofunction:: persistence_buf_byte
.. autofunction:: log_dir
.. autofunction:: logtostderr
.. autofunction:: logbuflevel
.. autofunction:: v
.. autofunction:: grpc_use_no_signal
.. autofunction:: collect_act_event
.. autofunction:: total_batch_num
.. autofunction:: default_data_type
.. autofunction:: default_initializer_conf
.. autofunction:: max_data_id_length
.. autofunction:: exp_run_conf
.. autofunction:: use_memory_allocation_algorithm_v2
.. autofunction:: enable_cudnn
.. autofunction:: cudnn_buf_limit_mbyte
.. autofunction:: cudnn_conv_force_fwd_algo
.. autofunction:: cudnn_conv_force_bwd_data_algo
.. autofunction:: cudnn_conv_force_bwd_filter_algo
.. autofunction:: enable_reused_mem
.. autofunction:: enable_inplace
.. autofunction:: enable_inplace_in_reduce_struct
.. autofunction:: enable_nccl
.. autofunction:: use_nccl_inter_node_communication
.. autofunction:: enable_all_reduce_group
.. autofunction:: all_reduce_group_num
.. autofunction:: all_reduce_group_min_mbyte
.. autofunction:: all_reduce_lazy_ratio
.. autofunction:: all_reduce_group_size_warmup
.. autofunction:: all_reduce_fp16
.. autofunction:: use_boxing_v2
.. autofunction:: enable_non_distributed_optimizer
.. autofunction:: disable_all_reduce_sequence
.. autofunction:: non_distributed_optimizer_group_size_mbyte
.. autofunction:: enable_true_half_config_when_conv
.. autofunction:: enable_float_compute_for_half_gemm
.. autofunction:: enable_auto_mixed_precision
.. autofunction:: concurrency_width

Training
----------------------------------
.. currentmodule:: oneflow.config.train
.. automodule:: oneflow.config.train
    :members:
    :undoc-members:
.. autofunction:: model_update_conf
.. autofunction:: loss_scale_factor
.. autofunction:: primary_lr
.. autofunction:: secondary_lr
.. autofunction:: weight_l1
.. autofunction:: bias_l1
.. autofunction:: weight_l2
.. autofunction:: bias_l2

Memory allocation
----------------------------------
.. currentmodule:: oneflow.config.static_mem_alloc_algo_white_list
.. automodule:: oneflow.config.static_mem_alloc_algo_white_list
    :members:
    :undoc-members:
.. autofunction:: show

.. currentmodule:: oneflow.config.static_mem_alloc_policy_white_list
.. automodule:: oneflow.config.static_mem_alloc_policy_white_list
    :members:
    :undoc-members:
.. autofunction:: has
.. autofunction:: add
.. autofunction:: remove
.. autofunction:: policy_mem_size_first
.. autofunction:: policy_mutual_exclusion_first
.. autofunction:: policy_time_line
