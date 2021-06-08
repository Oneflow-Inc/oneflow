import oneflow as flow
from typing import List, Tuple
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import Tensor

class Stack(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        assert isinstance(inputs, (List, Tuple))
        input_shape = inputs[0].shape
        max_dim = len(input_shape)

        # The axis must be in range [-(_max_dim +1), _max_dim]
        if self.dim < 0:
            self.dim = self.dim + max_dim + 1
        assert (self.dim >= 0) and (self.dim <= max_dim)
        input_list_length = len(inputs)
        for i in range(input_list_length):
            current_shape = inputs[i].shape
            assert (
                    input_shape == current_shape
            ), "Each tensor should have the same shape ! Found a tensor instance shape is: {}".format(
                current_shape
            )
            inputs[i] = flow.experimental.unsqueeze(inputs[i], dim=self.dim)
        return flow.experimental.cat(inputs, dim=self.dim)

@oneflow_export("stack")
@register_tensor_op("stack")
@experimental_api
def stack(inputs: Tensor, dim: int = 0) -> None:
    r"""Concatenates a sequence of tensors along a new dimension.
    The returned tensor shares the same underlying data with input tensors.
    A :attr:`dim` value within the range `[-input.ndimension() - 1, input.ndimension() + 1)`
    can be used. Negative :attr:`dim` will correspond to :meth:`stack`
    applied at :attr:`dim` = ``dim + input.ndimension() + 1``.
    Args:
        inputs (List[Tensor]): the list of input tensors. Each tensor should have the same shape.
        dim (int): the index at which to insert the singleton dimension
    For example:
    .. code-block:: python
        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()
        >>> x = flow.Tensor(np.random.rand(2, 4, 6))
        >>> y = flow.Tensor(np.random.rand(2, 4, 6))
        >>> out = flow.experimental.stack([x, y])
        >>> out.shape
        flow.Size([4, 4, 6])
    """
    return Stack(dim)(inputs)
