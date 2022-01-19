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


def index_select_op(input, dim, index):
    r"""The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchindex_select

    Select values along an axis specified by `dim`.

    :attr:`index` must be an Int32 Tensor with 1-D.
    :attr:`dim` must be in the range of input Dimensions.
    value of :attr:`index` must be in the range of the dim-th of input.
    Note that ``input`` and ``index`` do not broadcast against each other.  
    
    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (Tensor): the 1-D tensor containing the indices to index
    
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
    assert len(index.shape) == 1, "Dimensions of index should be 1-D"
    assert (
        dim < len(input.shape) and dim >= 0
    ), "Value of dim is out of range(dim should be in the range of [0, input dimensions) )"
    assert _input_args_is_int(
        index.tolist()
    ), "input index parameter is not legal!(index should be an 1-D int tensor)"
    index_rshp = list(input.shape)

    for index_i in index:
        assert (
            index_i < index_rshp[dim]
        ), "index is out of range(index shuold be lower than the dim-th dimension of input)"

    index_rshp[dim] = 1
    index_gather = index[0].expand(*index_rshp)
    if index.size()[0] > 1:
        for index_i in index[1:]:
            x = index_i.expand(*index_rshp)
            index_gather = flow.cat((index_gather, x), dim)

    return flow.gather(input, dim, index_gather)


@register_tensor_op("index_select")
def index_select_op_tensor(input, dim, index):
    """
    input.index_select(dim, index) -> Tensor
    See :func:`oneflow.index_select`
    """

    return index_select_op(input, dim, index)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
