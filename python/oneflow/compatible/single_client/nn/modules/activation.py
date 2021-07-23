from oneflow.compatible import single_client as flow
import oneflow._oneflow_internal
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from typing import Optional


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    if dim_num == 1:
        return (False, None)
    if axis < 0:
        axis += dim_num
    assert axis >= 0
    assert axis < dim_num
    need_transpose = False
    permute = list(range(dim_num))
    if axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return (need_transpose, permute)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
