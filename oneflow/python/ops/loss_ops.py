from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
from typing import Optional


@oneflow_export("smooth_l1_loss")
def smooth_l1_loss(
    prediction: remote_blob_util.BlobDef,
    label: remote_blob_util.BlobDef,
    beta: float = 1.0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SmoothL1Loss_")
        )
        .Op("smooth_l1_loss")
        .Input("prediction", [prediction])
        .Input("label", [label])
        .Output("loss")
    )
    op.Attr("beta", float(beta), "AttrTypeFloat")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]
