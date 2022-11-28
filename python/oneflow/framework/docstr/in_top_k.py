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
    oneflow.in_top_k,
    """
    in_top_k(targets, predictions, k) -> Tensor

    Says whether the targets are in the top K predictions.

    Args:
        targets (Tensor): the target tensor of type int32 or int64.
        predictions (Tensor): the predictions tensor of type float32 .
        k (int): Number of top elements to look at for computing precision.

    Returns:
        oneflow.Tensor: A Tensor of type bool. Computed Precision at k as a bool Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> targets1 = flow.tensor(np.array([3, 1]), dtype=flow.int32)
        >>> predictions1 = flow.tensor(np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],]), dtype=flow.float32)
        >>> out1 = flow.in_top_k(targets1, predictions1, k=1)
        >>> out1
        tensor([ True, False], dtype=oneflow.bool)
        >>> out2 = flow.in_top_k(targets1, predictions1, k=2)
        >>> out2
        tensor([True, True], dtype=oneflow.bool)
        >>> targets2 = flow.tensor(np.array([3, 1]), dtype=flow.int32, device=flow.device('cuda'))
        >>> predictions2 = flow.tensor(np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],]), dtype=flow.float32, device=flow.device('cuda'))
        >>> out3 = flow.in_top_k(targets2, predictions2, k=1)
        >>> out3
        tensor([ True, False], device='cuda:0', dtype=oneflow.bool)
    """,
)
