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
    oneflow.nn.functional.fold,
    r"""
    fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.fold.html.
    
    Combines an array of sliding local blocks into a large containing tensor.

    .. warning::
        Currently, only 3-D input tensors (batched image-like tensors) are supported, and only unbatched (3D) 
        or batched (4D) image-like output tensors are supported.

    See :class:`oneflow.nn.Fold` for details.
    """,
)

add_docstr(
    oneflow.nn.functional.unfold,
    r"""
    unfold(input, kernel_size, dilation=1, padding=0, stride=1)

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.unfold.html.

    Extracts sliding local blocks from a batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`oneflow.nn.Unfold` for details.
    """,
)
