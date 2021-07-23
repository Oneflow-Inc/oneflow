import os
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.distribute as distribute_util
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util
from typing import Optional, Union
import oneflow._oneflow_internal


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
