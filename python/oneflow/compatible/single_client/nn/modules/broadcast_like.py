from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module


class BroadCastLike(Module):
    def __init__(self, broadcast_axes: None) -> None:
        super().__init__()
        self.broadcast_axes = broadcast_axes

    def forward(self, x, like_tensor):
        return flow.F.broadcast_like(x, like_tensor, broadcast_axes=self.broadcast_axes)
