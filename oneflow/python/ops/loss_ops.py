from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("smooth_l1_loss")
def smooth_l1_loss(prediction, label, beta=1.0, name=None):
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
