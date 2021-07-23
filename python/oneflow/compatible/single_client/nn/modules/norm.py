from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op


class Norm(Module):
    def __init__(self, ord=None, dim=None, keepdim=False) -> None:
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def _vector_norm(self, x, ord, dim):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            raise ValueError("Norm order {} is not supported for vectors".format(ord))
        elif isinstance(ord, float) and ord in [float("inf"), float("-inf")]:
            if ord == float("inf"):
                return flow.experimental.max(flow.experimental.abs(x), dim=dim)
            else:
                return flow.experimental.min(flow.experimental.abs(x), dim=dim)
        elif isinstance(ord, int):
            if ord == 0:
                return flow.tensor([flow.experimental.argwhere(x).shape[0]])
            else:
                return flow.experimental.pow(
                    flow.experimental.sum(
                        flow.experimental.pow(flow.experimental.abs(x), ord), dim=dim
                    ),
                    1.0 / ord,
                )
        else:
            raise ValueError("Invalid norm order: {}".format(ord))

    def _matrix_norm(self, x, ord, dim):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            if ord == "nuc":
                raise NotImplementedError
            else:
                return flow.experimental.sqrt(
                    flow.experimental.sum(flow.experimental.square(x), dim=dim)
                )
        elif isinstance(ord, float) and ord in [float("inf"), float("-inf")]:
            if ord == float("inf"):
                return flow.experimental.max(
                    flow.experimental.sum(flow.experimental.abs(x), dim=1)
                )
            else:
                return flow.experimental.min(
                    flow.experimental.sum(flow.experimental.abs(x), dim=1)
                )
        elif isinstance(ord, int):
            if ord == 1:
                return flow.experimental.max(
                    flow.experimental.sum(flow.experimental.abs(x), dim=0)
                )
            elif ord == -1:
                return flow.experimental.min(
                    flow.experimental.sum(flow.experimental.abs(x), dim=0)
                )
            elif ord == 2:
                raise NotImplementedError
            elif ord == -2:
                raise NotImplementedError
            else:
                raise ValueError(
                    "Norm order {} is not supported for matrices".format(ord)
                )
        else:
            raise ValueError("Invalid norm order: {}".format(ord))

    def _whether_keepdim(self, x):
        if self.keepdim == True and self.dim != None:
            return flow.experimental.unsqueeze(x, self.dim)
        else:
            return x

    def forward(self, x):
        num_axes = len(x.shape)
        if self.dim == None and self.ord == None:
            res = self._vector_norm(x.reshape((1, -1))[0], ord=2, dim=self.dim)
        elif self.dim == None and self.ord != None:
            assert (
                num_axes <= 2
            ), "input must be 1-D or 2-D when dim is None and ord is not None"
            res = (
                self._vector_norm(x, self.ord, self.dim)
                if num_axes == 1
                else self._matrix_norm(x, self.ord, self.dim)
            )
        elif isinstance(self.dim, (int, tuple, list)):
            if isinstance(self.dim, int):
                self.dim = self.dim if self.dim >= 0 else self.dim + num_axes
                assert 0 <= self.dim < num_axes, "dim out of range"
                res = self._vector_norm(
                    x, ord=2 if self.ord == None else self.ord, dim=self.dim
                )
            else:
                temp = list(self.dim) if isinstance(self.dim, tuple) else self.dim
                for i in range(len(temp)):
                    temp[i] = temp[i] if temp[i] >= 0 else temp[i] + num_axes
                    assert 0 <= temp[i] < num_axes, "dim out of range"
                self.dim = temp
                res = self._matrix_norm(
                    x, ord="fro" if self.ord == None else self.ord, dim=self.dim
                )
        else:
            raise ValueError("Invalid dimension: {}".format(self.dim))
        return self._whether_keepdim(res)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
