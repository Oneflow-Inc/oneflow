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
from oneflow.framework.tensor import Tensor, register_tensor_op


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


def index_select_op(input, dim, index, sparse_grad=False):
    r"""Select values along an axis specified by `dim`.

    :attr:`index` must be an Int32 Tensor with dimension=1.
    :attr:`dim` must be in the range of input Dimensions.
    value of :attr:`index` must be in the range of the dim(th) of input.
    :attr:`out` will have the same shape as :attr:`index`.
    Note that ``input`` and ``index`` do not broadcast against each other.  
    
    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to select
    
    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,3],[4,5,6]], dtype=flow.int32)
        >>> input 
        tensor([[1, 2, 3],
                [4, 5, 6]], dtype=oneflow.int32)
        >>> index = flow.tensor([0,1], dtype=flow.int32)
        >>> output = flow.index_select(input, 1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
        >>> output = input.index_select(1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
    """
    assert sparse_grad is False, "Only support bool = False for now!"
    assert len(index.shape) == 1, "Dimensions of index should be an LongTensor"
    assert dim < len(input.shape) and dim > -1, "Value of dim is out of range"
    assert _input_args_is_int(index.tolist()), "input index parameter is not illegal!"
    index_rshp = list(input.shape)

    for index_i in index:
        assert (
            index_i < index_rshp[dim]
        ), "value of index out of range(index shuold lower than the first dimension of input)"

    index_rshp[dim] = 1
    index_gather = index[0].expand(index_rshp)
    if index.size()[0] > 1:
        for index_i in index[1:]:
            x = index_i.expand(index_rshp)
            index_gather = flow.cat((index_gather, x), dim)

    return flow.gather(input, index_gather, dim, sparse_grad)


@register_tensor_op("index_select")
def index_select_op_tensor(input, dim, index, sparse_grad=False):
    """
    input.index_select(dim, index, sparse_grad) -> Tensor
    See :func:`oneflow.index_select`
    """
    assert sparse_grad is False, "Only support bool = False for now!"
    assert len(index.shape) == 1, "Dimensions of index should be an LongTensor"
    assert dim < len(input.shape) and dim > -1, "Value of dim is out of range"
    assert _input_args_is_int(index.tolist()), "input index parameter is not illegal!"
    index_rshp = list(input.shape)

    for index_i in index:
        assert (
            index_i < index_rshp[dim]
        ), "value of index out of range(index shuold lower than the first dimension of input)"

    index_rshp[dim] = 1
    index_gather = index[0].expand(index_rshp)
    if index.size()[0] > 1:
        for index_i in index[1:]:
            x = index_i.expand(index_rshp)
            index_gather = flow.cat((index_gather, x), dim)

    return flow.gather(input, index_gather, dim, sparse_grad)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
