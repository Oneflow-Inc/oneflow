from typing import Tuple

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util


def ssp_variable_proxy(
    var: oneflow._oneflow_internal.BlobDesc, buffer_size: int = 1, name=None
) -> Tuple[oneflow._oneflow_internal.BlobDesc, oneflow._oneflow_internal.BlobDesc]:
    """ return ref_blob, value_blob """
    if name is None:
        name = id_util.UniqueStr("SspVariableProxy_")
    blob_dict = (
        flow.user_op_builder(name)
        .Op("ssp_variable_proxy")
        .Input("var", [var])
        .Output("ref")
        .Output("value")
        .Attr("buffer_size", buffer_size)
        .Build()
        .InferAndTryRun()
        .RemoteBlobDict()
    )
    return (blob_dict["ref"][0], blob_dict["value"][0])
