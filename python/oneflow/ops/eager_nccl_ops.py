from typing import Optional

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util


def eager_nccl_all_reduce(
    x: oneflow._oneflow_internal.BlobDesc,
    parallel_conf: str,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("EagerNcclAllReduce_")
        )
        .Op("eager_nccl_all_reduce")
        .Input("in", [x])
        .Output("out")
        .Attr("parallel_conf", parallel_conf)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
