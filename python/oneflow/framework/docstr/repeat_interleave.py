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
    oneflow.repeat_interleave,
    """
    repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor

    Repeat elements of a tensor.

    .. warning::

        This is different from :meth:`oneflow.Tensor.repeat` but similar to ``numpy.repeat``.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.repeat_interleave.html

    Args:
        input (oneflow.Tensor): the input Tensor.
        repeats (Tensor or int): The number of repetitions for each element.
            repeats is broadcasted to fit the shape of the given axis.
        dim (int, optional): The dimension along which to repeat values.
            By default, use the flattened input array, and return a flat output
            array.

    Keyword args:
        output_size (int, optional): Total output size for the given axis
            ( e.g. sum of repeats). If given, it will avoid stream syncronization
            needed to calculate output shape of the tensor.

    Returns:
        oneflow.Tensor: Repeated tensor which has the same shape as input, except along the given axis.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> y = flow.tensor([[1, 2], [3, 4]])
        >>> flow.repeat_interleave(y, 2)
        tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=oneflow.int64)
        >>> flow.repeat_interleave(y, 3, dim=1)
        tensor([[1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4]], dtype=oneflow.int64)
        >>> flow.repeat_interleave(y, flow.tensor([1, 2]), dim=0)
        tensor([[1, 2],
                [3, 4],
                [3, 4]], dtype=oneflow.int64)
        >>> flow.repeat_interleave(y, flow.tensor([1, 2]), dim=0, output_size=3)
        tensor([[1, 2],
                [3, 4],
                [3, 4]], dtype=oneflow.int64)
    
    ..
        Feature Stage of Operator [repeat_interleave].
        - Maintainer List [@BBuf]
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
            - Exception Message and Hint must be provided [Yes]
        - Beta Stage Check List [ ]
          - API(High compatibility with PyTorch 1.11, shouldn't have anything incompatible for a naive reason.)[Yes]
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
