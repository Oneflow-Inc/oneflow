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
import os
from typing import Optional, Union

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import distribute as distribute_util
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework import remote_blob as remote_blob_util
from oneflow.core.operator import op_conf_pb2 as op_conf_util
from oneflow.core.register import logical_blob_id_pb2 as logical_blob_id_util


def distributed_partial_fc_sample(
    weight: oneflow._oneflow_internal.BlobDesc,
    label: oneflow._oneflow_internal.BlobDesc,
    num_sample: int,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    parallel_num = flow.current_scope().device_parallel_desc_symbol.parallel_num
    assert num_sample % parallel_num == 0
    assert weight.shape[0] % parallel_num == 0
    return (
        flow.user_op_builder(
            name
            if name is not None
            else id_util.UniqueStr("DistributedPartialFcSample_")
        )
        .Op("distributed_partial_fc_sample")
        .Input("weight", [weight])
        .Input("label", [label])
        .Attr("num_sample", num_sample)
        .Output("mapped_label")
        .Output("sampled_label")
        .Output("sampled_weight")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
