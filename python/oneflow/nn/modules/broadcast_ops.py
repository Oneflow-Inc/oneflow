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
from oneflow.nn.modules.utils import _single, _handle_size_arg


def broadcast_shapes(*shapes):
    r"""broadcast_shapes(*shapes) -> Size

    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.broadcast_shapes.html.

    Similar to :func:`oneflow.broadcast_tensors` but for shapes.

    This is equivalent to ``flow.broadcast_tensors(*map(flow.empty, shapes))[0].shape``
    but avoids the need create to intermediate tensors.
    This is useful for broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Args:
        \*shapes (flow.Size): Shapes of tensors.

    Returns:
        A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.

    Example::

        >>> import oneflow as flow
        >>> flow.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        oneflow.Size([1, 3, 2])
    """
    shapes = _single(shapes)
    return flow._C.broadcast_shapes(shapes)


def broadcast_tensors(*tensors):
    r"""broadcast_tensors(*tensors) -> List of Tensors

    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.broadcast_tensors.html.

    Broadcasts the given tensors according to ``broadcasting-semantics``.

    Args:
        *tensors: any number of tensors of the same type

    .. warning::

        More than one element of a broadcasted tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> import oneflow as flow
        >>> x = flow.arange(3).view(1, 3)
        >>> y = flow.arange(2).view(2, 1)
        >>> a, b = flow.broadcast_tensors(x, y)
        >>> a.size()
        oneflow.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]], dtype=oneflow.int64)
    """
    tensors = _single(tensors)
    return flow._C.broadcast_tensors(tensors)


def broadcast_to(input, shape):
    r"""broadcast_to(input, shape) -> Tensors

    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.broadcast_to.html.

    Broadcasts ``input`` to the shape ``shape``. Equivalent to calling ``input.expand(shape)``. See :func:`oneflow.expand` for details.

    Args:
        input (oneflow.Tensor): the input tensor.
        shape (list, tuple, or oneflow.Size): the new shape.

    Example::

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> flow.broadcast_to(x, (3, 3))
        tensor([[1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]], dtype=oneflow.int64)
    """
    shape = _handle_size_arg(shape)
    shape = _single(shape)
    return flow._C.broadcast_to(input, shape)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
