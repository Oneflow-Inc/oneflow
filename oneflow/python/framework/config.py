from __future__ import absolute_import

import oneflow.python.framework.config_util as config_util

## for the 'main' function
## resource_conf
ctrl_port = config_util.MakeResourceConfigDecorator("ctrl_port", int)
data_port = config_util.MakeResourceConfigDecorator("data_port", int)
gpu_device_num = config_util.MakeResourceConfigDecorator("gpu_device_num", int)
cpu_device_num = config_util.MakeResourceConfigDecorator("cpu_device_num", int)
comm_net_worker_num = config_util.MakeResourceConfigDecorator("comm_net_worker_num", int)
max_mdsave_worker_num = config_util.MakeResourceConfigDecorator("max_mdsave_worker_num", int)
use_rdma = config_util.MakeResourceConfigDecorator("use_rdma", bool)
rdma_mem_block_mbyte = config_util.MakeResourceConfigDecorator("rdma_mem_block_mbyte", int)
rdma_recv_msg_buf_mbyte = config_util.MakeResourceConfigDecorator("rdma_recv_msg_buf_mbyte", int)
reserved_host_mem_mbyte = config_util.MakeResourceConfigDecorator("reserved_host_mem_mbyte", int)
reserved_device_mem_mbyte = config_util.MakeResourceConfigDecorator("reserved_device_mem_mbyte", int)
## io_conf
model_load_snapshot_path = config_util.MakeIOConfigDecorator("model_load_snapshot_path", str)
model_save_snapshots_path = config_util.MakeIOConfigDecorator("model_save_snapshots_path", str)
enable_write_snapshot = config_util.MakeIOConfigDecorator("enable_write_snapshot", bool)
save_downloaded_file_to_local_fs = \
    config_util.MakeIOConfigDecorator("save_downloaded_file_to_local_fs", bool)
persistence_buf_byte = config_util.MakeIOConfigDecorator("persistence_buf_byte", int)
## cpp_flags_conf
log_dir = config_util.MakeCppFlagsConfigDecorator("log_dir", str)
logtostderr = config_util.MakeCppFlagsConfigDecorator("logtostderr", int)
logbuflevel = config_util.MakeCppFlagsConfigDecorator("logbuflevel", int)
v = config_util.MakeCppFlagsConfigDecorator("v", int)
grpc_use_no_signal = config_util.MakeCppFlagsConfigDecorator("grpc_use_no_signal", int)
## profiler_conf
collect_act_event = config_util.MakeProfilerConfigDecorator("collect_act_event", bool)

## for the 'remote' function
default_data_type = config_util.MakeJobOtherConfigDecorator("default_data_type", int)
data_part_num = config_util.MakeJobOtherConfigDecorator("data_part_num", int)
enable_cudnn = config_util.MakeJobOtherConfigDecorator("enable_cudnn", bool)
cudnn_buf_limit_mbyte = config_util.MakeJobOtherConfigDecorator("cudnn_buf_limit_mbyte", int)
enable_mem_sharing = config_util.MakeJobOtherConfigDecorator("enable_mem_sharing", bool)
enable_inplace = config_util.MakeJobOtherConfigDecorator("enable_inplace", bool)
enable_nccl = config_util.MakeJobOtherConfigDecorator("enable_nccl", bool)
use_nccl_inter_node_communication = \
    config_util.MakeJobOtherConfigDecorator("use_nccl_inter_node_communication", bool)
all_reduce_group_num = config_util.MakeJobOtherConfigDecorator("all_reduce_group_num", int)
all_reduce_lazy_ratio = config_util.MakeJobOtherConfigDecorator("all_reduce_lazy_ratio", float)
all_reduce_group_min_mbyte = \
    config_util.MakeJobOtherConfigDecorator("all_reduce_group_min_mbyte", int)
all_reduce_group_size_warmup = \
    config_util.MakeJobOtherConfigDecorator("all_reduce_group_size_warmup", float)
all_reduce_fp16 = config_util.MakeJobOtherConfigDecorator("all_reduce_fp16", bool)
concurrency_width = config_util.MakeJobOtherConfigDecorator("concurrency_width", int)
