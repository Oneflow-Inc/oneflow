oneflow.nn.Graph
============================================================
Base class for running neural networks in Static Graph Mode.
------------------------------------------------------------
.. currentmodule:: oneflow.nn
.. autoclass:: oneflow.nn.Graph
    :members: __init__,
            build,
            __call__,
            add_optimizer,
            set_grad_scaler,
            state_dict,
            load_state_dict,
            name,
            debug,
            __repr__,
    :member-order: bysource



.. autoclass:: oneflow.nn.graph.graph_config.GraphConfig
    :members: enable_amp,
            allow_fuse_model_update_ops,
            allow_fuse_add_to_output,
            allow_fuse_cast_scale,
            set_gradient_accumulation_steps,
            set_zero_redundancy_optimizer_mode,
            set_zero_redundancy_optimizer_min_size_after_split,
            enable_xla_jit,
            enable_tensorrt,
            enable_openvino,
            enable_cudnn_conv_heuristic_search_algo,
    :member-order: bysource



.. autoclass:: oneflow.nn.graph.block_config.BlockConfig
    :members: stage_id,
            activation_checkpointing,
    :member-order: bysource

