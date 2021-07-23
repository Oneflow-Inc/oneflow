from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from typing import Optional, Sequence


class Transpose(Module):
    def __init__(
        self, dim0, dim1, conjugate: bool = False, batch_axis_non_change: bool = False
    ) -> None:
        super().__init__()
        if conjugate:
            raise NotImplementedError
        if batch_axis_non_change:
            raise NotImplementedError
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x_shape = x.shape
        dim0 = self.dim0
        dim1 = self.dim1
        if dim0 < 0:
            dim0 += len(x_shape)
        if dim1 < 0:
            dim1 += len(x_shape)
        assert dim0 >= 0 and dim0 < len(
            x_shape
        ), "Invalid dim0 {}, len(shape): {}".format(dim0, len(x_shape))
        assert dim1 >= 0 and dim1 < len(
            x_shape
        ), "Invalid dim1 {}, len(shape): {}".format(dim1, len(x_shape))
        perm = []
        for i in range(len(x_shape)):
            perm.append(i)
        (perm[dim0], perm[dim1]) = (perm[dim1], perm[dim0])
        return flow.F.transpose(x, perm=perm)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
