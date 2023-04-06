"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module
from typing import Tuple, Union, List


class Flatten(Module):
    """Flattens a contiguous range of dims into a tensor. For use with: nn.Sequential.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    

    For example: 

    .. code-block:: python 

        >>> import oneflow as flow
        >>> input = flow.Tensor(32, 1, 5, 5)
        >>> m = flow.nn.Flatten()
        >>> output = m(input)
        >>> output.shape
        oneflow.Size([32, 25])

    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return flow._C.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


class Unflatten(Module):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it is a `int` type

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `oneflow.Size` for `Tensor` input

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`

    Args:
        dim (int): Dimension to be unflattened
        unflattened_size (Union[oneflow.Size, Tuple, List]): New shape of the unflattened dimension

    Examples:
        >>> input = oneflow.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([2, 2, 5, 5])
        >>> # With oneflow.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, oneflow.Size([2, 5, 5]))
        >>> )
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([2, 2, 5, 5])
    """

    def __init__(
        self, dim: int, unflattened_size: Union[Tuple[int], List[int], flow.Size]
    ) -> None:
        super(Unflatten, self).__init__()

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
        else:
            raise TypeError("invalid argument type for dim parameter, expected `int`")

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_int(self, input):
        if isinstance(input, (tuple, list)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "
                        + "but found element of type {} at pos {}".format(
                            type(elem).__name__, idx
                        )
                    )
            return
        raise TypeError(
            "unflattened_size must be a tuple of ints, but found type {}".format(
                type(input).__name__
            )
        )

    def forward(self, input):
        return flow._C.unflatten(input, self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        return "dim={}, unflattened_size={}".format(self.dim, self.unflattened_size)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
