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


def norm(input, p="fro", dim=None, keepdim=False, dtype=None):
    """
    Returns the matrix norm or vector norm of a given tensor.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.norm.html.

    .. warning::

        Use :func:`oneflow.linalg.norm`, instead, or :func:`oneflow.linalg.vector_norm`
        when computing vector norms and :func:`oneflow.linalg.matrix_norm` when
        computing matrix norms. Note, however, the signature for these functions
        is slightly different than the signature for oneflow.norm.

    Args:
        input (Tensor): The input tensor. Its data type must be either a floating
            point or complex type. For complex inputs, the norm is calculated using the
            absolute value of each element. If the input is complex and neither
            :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is
            complexfloat).

        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            ======  ==============  ==========================
            ord     matrix norm     vector norm
            ======  ==============  ==========================
            'fro'   Frobenius norm  --
            'nuc'   nuclear norm    --
            Number  --              sum(abs(x)**p)**(1./p)
            ======  ==============  ==========================

            The vector norm can be calculated across any number of dimensions.
            The corresponding dimensions of :attr:`input` are flattened into
            one dimension, and the norm is calculated on the flattened
            dimension.

            Frobenius norm produces the same result as ``p=2`` in all cases
            except when :attr:`dim` is a list of three or more dims, in which
            case Frobenius norm throws an error.

            Nuclear norm can only be calculated across exactly two dimensions.

        dim (int, tuple of ints, list of ints, optional):
            Specifies which dimension or dimensions of :attr:`input` to
            calculate the norm across. If :attr:`dim` is ``None``, the norm will
            be calculated across all dimensions of :attr:`input`. If the norm
            type indicated by :attr:`p` does not support the specified number of
            dimensions, an error will occur.
        keepdim (bool, optional): whether the output tensors have :attr:`dim`
            retained or not. Ignored if :attr:`dim` = ``None`` and
            :attr:`out` = ``None``. Default: ``False``
        dtype (:class:`oneflow.dtype`, optional): the desired data type of
            returned tensor. If specified, the input tensor is casted to
            :attr:`dtype` while performing the operation. Default: None.

    .. note::
        Even though ``p='fro'`` supports any number of dimensions, the true
        mathematical definition of Frobenius norm only applies to tensors with
        exactly two dimensions. :func:`oneflow.linalg.norm` with ``ord='fro'`` aligns
        with the mathematical definition, since it can only be applied across
        exactly two dimensions.

    Example::

        >>> import oneflow as flow
        >>> a = flow.arange(9, dtype= flow.float) - 4
        >>> b = a.reshape((3, 3))
        >>> flow.norm(a)
        tensor(7.7460, dtype=oneflow.float32)
        >>> flow.norm(b)
        tensor(7.7460, dtype=oneflow.float32)
        >>> flow.norm(a, float('inf'))
        tensor(4., dtype=oneflow.float32)
        >>> flow.norm(b, float('inf'))
        tensor(9., dtype=oneflow.float32)
        >>> c = flow.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= flow.float)
        >>> flow.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000], dtype=oneflow.float32)
        >>> flow.norm(c, dim=1)
        tensor([3.7417, 4.2426], dtype=oneflow.float32)
        >>> flow.norm(c, p=1, dim=1)
        tensor([6., 6.], dtype=oneflow.float32)
        >>> d = flow.arange(8, dtype= flow.float).reshape(2,2,2)
        >>> flow.norm(d, dim=(1,2))
        tensor([ 3.7417, 11.2250], dtype=oneflow.float32)
        >>> flow.norm(d[0, :, :]), flow.norm(d[1, :, :])
        (tensor(3.7417, dtype=oneflow.float32), tensor(11.2250, dtype=oneflow.float32))
    """
    if type(p) == str or dim != None:
        return flow._C.norm(input=input, ord=p, dim=dim, keepdim=keepdim, dtype=dtype)
    return flow._C.norm(
        input=input, ord=p, dim=dim, keepdim=keepdim, dtype=dtype, for_norm=True
    )
