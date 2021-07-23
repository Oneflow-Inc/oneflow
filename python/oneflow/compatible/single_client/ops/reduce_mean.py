import collections
from typing import Optional, Union

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.core.operator import op_conf_pb2 as op_conf_util
from oneflow.compatible.single_client.core.register import (
    logical_blob_id_pb2 as logical_blob_id_util,
)
from oneflow.compatible.single_client.python.framework import id_util as id_util
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)


def reduce_mean(
    input_blob: oneflow._oneflow_internal.BlobDesc,
    axis: Optional[Union[collections.Sized, int]] = None,
    keepdims: bool = False,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    """This operator computes the mean of input Blob along the specified axis

    Args:
        input_blob (oneflow._oneflow_internal.BlobDesc): A Blob
        axis (Optional[Union[collections.Sized, int]], optional): The dimension along which the mean value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Blob. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result of average on the specified axis of input Blob

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import numpy as np
        import oneflow.compatible.single_client.typing as tp


        @flow.global_function()
        def reduce_mean_Job(x: tp.Numpy.Placeholder((3, 3))
        ) -> tp.Numpy:
            return flow.math.reduce_mean(x, axis=1, keepdims=True)


        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        out = reduce_mean_Job(x)

        # out [[2.]
        #      [5.]
        #      [8.]]

    """
    reduce_sum = flow.math.reduce_sum(
        input_blob, axis=axis, keepdims=keepdims, name=name
    )
    if input_blob.is_dynamic:
        reduce_count = flow.math.reduced_shape_elem_cnt(
            input_blob, axis=axis, dtype=input_blob.dtype
        )
        return reduce_sum / reduce_count
    else:
        if axis is None:
            axes = []
        else:
            axes = list(axis) if isinstance(axis, collections.Sized) else [axis]
        reduce_count = 1
        if len(axes) == 0:
            for dim in input_blob.shape:
                reduce_count *= dim
        else:
            for i in axes:
                reduce_count *= input_blob.shape[i]
        return flow.math.multiply(reduce_sum, 1.0 / reduce_count)
