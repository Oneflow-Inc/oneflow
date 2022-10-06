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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.index_select,
    """
    input.index_select(dim, index) -> Tensor

    Select values along an axis specified by `dim`.

    :attr:`index` must be an Int32 Tensor with 1-D.
    :attr:`dim` must be in the range of input Dimensions.
    value of :attr:`index` must be in the range of the dim-th of input.
    Note that ``input`` and ``index`` do not broadcast against each other.  
    
    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.index_select.html.

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
        >>> index = flow.tensor([0,1], dtype=flow.int64)
        >>> output = flow.index_select(input, 1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
        >>> output = input.index_select(1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
    
    ..
        Feature Stage of Operator [index_select].
        - Maintainer List [@QiangX-man, @hjchen2, @strint]
        - Current Stage [ ]
        - Alpha Stage Check List [ ]
          - API(Compatible with PyTorch 1.11, anything incompatible must be noted in API Doc.)[Yes]
          - Doc(API Doc must be provided and showed normally on the web page.)[Yes]
          - Functionality and its' Test [ ]
            - Functionality is highly compatiable with PyTorch 1.11. [Yes]
            - eager local [Yes] [@QiangX-man, @hjchen2]
              - forward [Yes]
              - backward [Yes]
              - gpu [Yes]
              - cpu [Yes]
            - graph local [ ] [@BBuf, @strint, @hjchen2]
              - forward [Yes]
              - backward [ ]
              - gpu [Yes]
              - cpu [Yes]
          - Exception Handling
            - Exception Message and Hint must be provided [ ]
        - Beta Stage Check List [ ]
          - API(High compatibility with PyTorch 1.11, shouldn't have anything incompatible for a naive reason.)[ ]
          - Doc(Same standard as Alpha Stage)[ ]
          - Functionality and its' Test [ ]
            - eager global [ ]
              - forward [ ]
              - backward [ ]
              - gpu [ ]
              - cpu [ ]
            - graph gloal [ ]
              - forward [ ]
              - backward [ ]
              - gpu [ ]
              - cpu [ ]
          - Performance and Scalability(Must be evaluated.)[ ]
            - CUDA kernel [ ]
            - CPU kernel [ ]
            - N nodes M devices [ ]
          - Exception Handling [ ]
            - Exception Message and Hint must be provided [ ]
            - Try you best to do Exception Recovery [ ]
        - Stable Stage Check List [ ]
          - API(Same standard as Beta Stage)[ ]
          - Doc(Same standard as Beta Stage)[ ]
          - Functionality and its' Test [ ]
            - fp16 and AMP [ ]
            - NHWC [ ]
          - Performance and Scalability(Must be evaluated.)[ ]
          - Exception Handling [ ]
    """,
)
