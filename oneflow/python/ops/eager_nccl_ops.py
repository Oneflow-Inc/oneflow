from __future__ import absolute_import

from typing import Optional

import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("eager_nccl_all_reduce")
def eager_nccl_all_reduce(
    x: remote_blob_util.BlobDef, parallel_conf: str, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
