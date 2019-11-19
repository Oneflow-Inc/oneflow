from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


import collections


@oneflow_export("math.reduce_sum")
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    r"""Sum of elements across dimensions of a `Blob`.
    
    Analogous to `tf.math.reduce_sum <https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum>`_

    Args:
        input_tensor: A `Blob`.
        axis: Dimensions to reduce. By default, all dimensions will be reduced.
        keepdims: If true, every reduced dimension with a length of 1 will be kept.
        name: A name for the operator (optional).
    Returns:
        A `Blob`.
    """
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ReduceSum_")
    )
    setattr(op_conf.reduce_sum_conf, "in", input_tensor.logical_blob_name)
    setattr(op_conf.reduce_sum_conf, "out", "out")
    if axis is not None:
        op_conf.reduce_sum_conf.axis[:] = (
            list(axis) if isinstance(axis, collections.Sized) else [axis]
        )
    setattr(op_conf.reduce_sum_conf, "keep_dims", keepdims)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
