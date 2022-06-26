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
from oneflow.framework.config_util import (
    api_load_library as load_library,
    api_machine_num as machine_num,
    api_cpu_device_num as cpu_device_num,
    api_gpu_device_num as gpu_device_num,
    api_comm_net_worker_num as comm_net_worker_num,
    api_max_mdsave_worker_num as max_mdsave_worker_num,
    api_numa_aware_cuda_malloc_host as numa_aware_cuda_malloc_host,
    api_compute_thread_pool_size as compute_thread_pool_size,
    api_reserved_host_mem_mbyte as reserved_host_mem_mbyte,
    api_reserved_device_mem_mbyte as reserved_device_mem_mbyte,
    api_enable_debug_mode as enable_debug_mode,
    api_enable_legacy_model_io as enable_legacy_model_io,
    api_enable_model_io_v2 as enable_model_io_v2
)

from oneflow.utils.torch.from_or_to_torch_tensor import from_torch, to_torch
