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
    oneflow.linalg.inv,
    """linalg.inv(A) -> Tensor

    Computes the inverse of a square matrix if it exists.
    Throws a `RuntimeError` if the matrix is not invertible.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
    for a matrix :math:`A \in \mathbb{K}^{n \times n}`,
    its **inverse matrix** :math:`A^{-1} \in \mathbb{K}^{n \times n}` (if it exists) is defined as

    .. math::

        A^{-1}A = AA^{-1} = \mathrm{I}_n

    where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

    The inverse matrix exists if and only if :math:`A` is `invertible`_. In this case,
    the inverse is unique.

    Supports input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices, and if :attr:`A` is a batch of matrices
    then the output has the same batch dimensions.

    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of invertible matrices.

    Raises:
        RuntimeError: if the matrix :attr:`A` or any matrix in the batch of matrices :attr:`A` is not invertible.

    Examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> A = flow.tensor([[ 1.3408, -0.7788,  1.0551, -0.5866],
        ...                  [ 0.8480,  0.8350,  0.9781, -0.1297],
        ...                  [-0.0881, -0.6142, -0.3833,  0.3232],
        ...                  [ 1.2841,  0.7517, -0.3849,  0.2515]])
        >>> flow.linalg.inv(A)
        tensor([[ 0.3105, -0.0811,  0.1288,  0.5169],
        ...     [-0.3457,  0.1716, -0.7133,  0.1987],
        ...     [-0.0593,  1.1706,  0.8694, -0.6516],
        ...     [-0.6427,  1.6923,  2.8049, -0.2541]], dtype=oneflow.float32)

        >>> A = flow.tensor([[[ 0.6144,  0.1027, -0.1353],
        ...                   [-1.4415, -0.6731,  0.3723],
        ...                   [ 0.4069, -0.8940,  1.4056]],
        ...                  [[-1.1891, -0.3897, -1.5015],
        ...                   [ 0.3028,  1.1040,  0.2600],
        ...                   [-1.6970,  0.4238,  0.9146]]])
        >>> flow.linalg.inv(A)
        tensor([[[ 1.6830,  0.0644,  0.1449],
        ...      [-5.9755, -2.5206,  0.0925],
        ...      [-4.2879, -1.6219,  0.7283]],
        ...
        ...     [[-0.2370,  0.0737, -0.4100],
        ...      [ 0.1892,  0.9579,  0.0384],
        ...      [-0.5274, -0.3070,  0.3148]]], dtype=oneflow.float32)

    .. _invertible:
        https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
    
    ..
        Feature Stage of Operator [linalg.inv].
        - Maintainer List [@simonJJJ]
        - Current Stage [pre Alpha]
        - Alpha Stage Check List [ ]
          - API(Compatible with PyTorch 1.11, anything incompatible must be noted in API Doc.)[Yes]
          - Doc(API Doc must be provided and showed normally on the web page.)[Yes]
          - Functionality and its' Test [ ]
            - Functionality is highly compatiable with PyTorch 1.11. [Yes]
            - eager local [Yes] [@simonJJJ]
              - forward [Yes]
              - backward [Yes]
              - gpu [Yes]
              - cpu [Yes]
            - graph local [ ] [@simonJJJ]
              - forward [Yes]
              - backward [ ]
              - gpu [Yes]
              - cpu [Yes]
          - Exception Handling
            - Exception Message and Hint must be provided [Yes]
        - Beta Stage Check List [ ]
          - API(High compatibility with PyTorch 1.11, shouldn't have anything incompatible for a naive reason.)[ ]
          - Doc(Same standard as Alpha Stage)[Yes]
          - Functionality and its' Test [ ]
            - eager global [Yes] [@simonJJJ]
              - forward [Yes]
              - backward [Yes]
              - gpu [Yes]
              - cpu [Yes]
            - graph gloal [Yes]
              - forward [Yes]
              - backward [ ]
              - gpu [Yes]
              - cpu [Yes]
          - Performance and Scalability(Must be evaluated.)[ ]
            - CUDA kernel [ ]
            - CPU kernel [ ]
            - N nodes M devices [ ]
          - Exception Handling [Yes]
            - Exception Message and Hint must be provided [Yes]
            - Try you best to do Exception Recovery [Yes]
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
