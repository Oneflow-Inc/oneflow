from typing import List, Tuple

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import (
    Tensor,
    register_tensor_op,
)
from oneflow.compatible.single_client.python.nn.module import Module


class Stack(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        assert isinstance(inputs, (List, Tuple))
        input_shape = inputs[0].shape
        max_dim = len(input_shape)
        if self.dim < 0:
            self.dim = self.dim + max_dim + 1
        assert self.dim >= 0 and self.dim <= max_dim
        input_list_length = len(inputs)
        unsqueezed = list()
        for i in range(input_list_length):
            current_shape = inputs[i].shape
            assert (
                input_shape == current_shape
            ), "Each tensor should have the same shape ! Found a tensor instance shape is: {}".format(
                current_shape
            )
            unsqueezed.append(inputs[i].unsqueeze(dim=self.dim))
        return flow.experimental.cat(unsqueezed, dim=self.dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
