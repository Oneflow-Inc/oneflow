import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.interpret_util as interpret_util
import oneflow.framework.distribute as distribute_util
import oneflow.framework.id_util as id_util
import oneflow.framework.input_blob_def as input_blob_util
import oneflow.framework.remote_blob as remote_blob_util
import oneflow._oneflow_internal
from typing import Optional, Tuple


def indexed_slices_reduce_sum(
    indices: input_blob_util.ArgBlobDef,
    values: input_blob_util.ArgBlobDef,
    name: Optional[str] = None,
) -> Tuple[oneflow._oneflow_internal.BlobDesc]:
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("IndexedSlicesReduceSum_")
        )
        .Op("indexed_slices_reduce_sum")
        .Input("x_indices", [indices])
        .Input("x_values", [values])
        .Output("y_indices")
        .Output("y_values")
        .Output("num_unique")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()
