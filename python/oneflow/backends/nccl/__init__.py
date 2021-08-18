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
    api_enable_fusion as boxing_fusion_enable,
    api_nccl_fusion_all_reduce as boxing_fusion_all_reduce,
    api_nccl_fusion_all_reduce_use_buffer as boxing_fusion_all_reduce_use_buffer,
    api_nccl_fusion_reduce as boxing_fusion_reduce,
    api_nccl_fusion_broadcast as boxing_fusion_broadcast,
    api_nccl_fusion_all_gather as boxing_fusion_all_gather,
    api_nccl_fusion_reduce_scatter as boxing_fusion_reduce_scatter,
    api_nccl_enable_mixed_fusion as boxing_fusion_enable_mixed,
    api_nccl_fusion_threshold_mb as boxing_fusion_threshold_mb,
    api_nccl_fusion_max_ops as boxing_fusion_max_ops_num,
)

from oneflow.framework.config_util import (
    api_nccl_num_streams as streams_num,
)

from oneflow.framework.config_util import (
    api_nccl_enable_all_to_all as enable_all_to_all,
)

from oneflow.framework.config_util import api_nccl_use_compute_stream as reuse_compute_stream 