from typing import Optional, Sequence
import oneflow as flow
from oneflow.nn.module import Module


def _calc_broadcast_axes(x, like_tensor):
    num_prepend = len(like_tensor.shape) - len(x.shape)
    prepend_shape = [1] * num_prepend + list(x.shape)
    broadcast_axes = [x for x in range(num_prepend)]
    for i in range(num_prepend, len(prepend_shape)):
        if prepend_shape[i] != like_tensor.shape[i]:
            if prepend_shape[i] != 1:
                raise RuntimeError(
                    f"output with shape {x.shape} doesn't match the broadcast shape {like_tensor.shape}"
                )
            else:
                broadcast_axes.append(i)
    return tuple(broadcast_axes)


class BroadCastLike(Module):
    def __init__(self, broadcast_axes: Optional[Sequence] = None) -> None:
        super().__init__()
        self.broadcast_axes = broadcast_axes

    def forward(self, x, like_tensor):
        if self.broadcast_axes is None:
            broadcast_axes = _calc_broadcast_axes(x, like_tensor)
        else:
            broadcast_axes = self.broadcast_axes
        return flow.F.broadcast_like(x, like_tensor, broadcast_axes=broadcast_axes)


def broadcast_like_op(x, like_tensor, broadcast_axes: Optional[Sequence] = None):
    return BroadCastLike(broadcast_axes=broadcast_axes)(x, like_tensor)
