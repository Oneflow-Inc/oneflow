from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op

class Where(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, condition, x, y):
        assert condition.dtype == flow.int32 or condition.dtype == flow.int8
        if isinstance(x, int) or isinstance(x, float):
            x = flow.Tensor([float(x)], dtype=flow.float32, device=flow.device(condition.device.type))
        if isinstance(y, int) or isinstance(y, float):
            y = flow.Tensor([float(y)], dtype=flow.float32, device=flow.device(condition.device.type))
        assert condition.device.type == x.device.type and condition.device.type == y.device.type
        assert len(condition.shape) == len(x.shape) and len(condition.shape) == len(y.shape), f"The dim of where module's inputs can not match, please check!"
        broadcast_cond = condition
        broadcast_x = x
        broadcast_y = y
        broadcast_like_shape = []
        broadcast_condition_axes = []
        broadcast_x_axes = []
        broadcast_y_axes = []
        for i in range(len(x.shape)):
            max_dim = max(x.shape[i], max(y.shape[i], condition.shape[i]))
            broadcast_like_shape.append(max_dim)
            if max_dim != condition.shape[i]:
                broadcast_condition_axes.append(i)
            if max_dim != x.shape[i]:
                broadcast_x_axes.append(i)
            if max_dim != y.shape[i]:
                broadcast_y_axes.append(i)
        broadcast_like_tensor = flow.experimental.zeros(tuple(broadcast_like_shape), dtype=flow.float32)
        broadcast_like_tensor = broadcast_like_tensor.to(x.device.type)
        broadcast_like_tensor.requires_grad = x.requires_grad or y.requires_grad
        if len(broadcast_condition_axes) != 0:
            condition = flow.experimental.cast(condition, flow.float32)
            broadcast_cond = flow.experimental.broadcast_like(condition, broadcast_like_tensor, tuple(broadcast_condition_axes))
            broadcast_cond = flow.experimental.cast(broadcast_cond, flow.int32)
        if len(broadcast_x_axes) != 0:
            broadcast_x = flow.experimental.broadcast_like(x, broadcast_like_tensor, broadcast_axes=tuple(broadcast_x_axes))
        if len(broadcast_y_axes) != 0:
            broadcast_y = flow.experimental.broadcast_like(y, broadcast_like_tensor, broadcast_axes=tuple(broadcast_y_axes))
        return flow.F.where(broadcast_cond, broadcast_x, broadcast_y)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)