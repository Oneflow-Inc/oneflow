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
    api_nccl_fusion_threshold_mb as set_fusion_threshold_mbytes,
    api_nccl_fusion_max_ops as set_fusion_max_ops_num,
    api_nccl_fusion_all_reduce as allow_fuse_all_reduce,
    api_nccl_fusion_reduce_scatter as allow_fuse_reduce_scatter,
    api_nccl_fusion_all_gather as allow_fuse_all_gather,
    api_nccl_fusion_reduce as allow_fuse_reduce,
    api_nccl_fusion_broadcast as allow_fuse_broadcast,
    api_nccl_enable_mixed_fusion as allow_fuse_mixed_ops,
    api_nccl_fusion_all_reduce_use_buffer as enable_use_buffer_to_fuse_all_reduce,
)

from oneflow.framework.config_util import api_nccl_num_streams as set_stream_num

from oneflow.framework.config_util import (
    api_nccl_enable_all_to_all as enable_all_to_all,
)

from oneflow.framework.config_util import (
    api_nccl_use_compute_stream as enable_use_compute_stream,
)

from oneflow.framework.config_util import (
    api_disable_group_boxing_by_dst_parallel as disable_group_boxing_by_dst_parallel,
)
