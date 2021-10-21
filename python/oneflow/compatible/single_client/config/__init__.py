"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from oneflow.compatible.single_client.framework.config_util import (
    api_collect_act_event as collect_act_event,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_comm_net_worker_num as comm_net_worker_num,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_compute_thread_pool_size as compute_thread_pool_size,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_cpu_device_num as cpu_device_num,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_disable_group_boxing_by_dst_parallel as disable_group_boxing_by_dst_parallel,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_enable_debug_mode as enable_debug_mode,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_enable_legacy_model_io as enable_legacy_model_io,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_enable_mem_chain_merge as enable_mem_chain_merge,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_enable_model_io_v2 as enable_model_io_v2,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_enable_tensor_float_32_compute as enable_tensor_float_32_compute,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_gpu_device_num as gpu_device_num,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_legacy_model_io_enabled as legacy_model_io_enabled,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_load_library as load_library,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_load_library_now as load_library_now,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_machine_num as machine_num,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_max_mdsave_worker_num as max_mdsave_worker_num,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_nccl_use_compute_stream as nccl_use_compute_stream,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_numa_aware_cuda_malloc_host as enable_numa_aware_cuda_malloc_host,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_rdma_mem_block_mbyte as rdma_mem_block_mbyte,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_rdma_recv_msg_buf_mbyte as rdma_recv_msg_buf_mbyte,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_reserved_device_mem_mbyte as reserved_device_mem_mbyte,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_reserved_host_mem_mbyte as reserved_host_mem_mbyte,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_thread_enable_local_message_queue as thread_enable_local_message_queue,
)
from oneflow.compatible.single_client.framework.config_util import (
    api_use_rdma as use_rdma,
)

from . import collective_boxing
