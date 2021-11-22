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

from oneflow.compatible.single_client import unittest
from oneflow.compatible.single_client.experimental.indexed_slices_ops import (
    indexed_slices_reduce_sum,
)
from oneflow.compatible.single_client.experimental.interface_op_read_and_write import (
    FeedValueToInterfaceBlob as set_interface_blob_value,
)
from oneflow.compatible.single_client.experimental.interface_op_read_and_write import (
    GetInterfaceBlobValue as get_interface_blob_value,
)
from oneflow.compatible.single_client.experimental.namescope import (
    deprecated_name_scope as name_scope,
)
from oneflow.compatible.single_client.experimental.square_sum_op import square_sum
from oneflow.compatible.single_client.experimental.ssp_variable_proxy_op import (
    ssp_variable_proxy,
)
from oneflow.compatible.single_client.experimental.typing_check import (
    api_enable_typing_check as enable_typing_check,
)
from oneflow.compatible.single_client.experimental.unique_op import unique_with_counts
from oneflow.compatible.single_client.framework.c_api_util import (
    GetJobSet as get_job_set,
)
from oneflow.compatible.single_client.ops.array_ops import (
    logical_slice,
    logical_slice_assign,
)
from oneflow.compatible.single_client.ops.assign_op import (
    api_one_to_one_assign as eager_assign_121,
)
from oneflow.compatible.single_client.ops.util.custom_op_module import (
    CustomOpModule as custom_op_module,
)

from . import scope
