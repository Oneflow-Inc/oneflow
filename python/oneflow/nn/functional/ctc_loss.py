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
from oneflow.framework.tensor import Tensor
import oneflow as flow


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank=0,
    reduction="mean",
    zero_infinity=False,
) -> Tensor:
    r"""
    The Connectionist Temporal Classification loss.
    
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html

    See :class:`~oneflow.nn.CTCLoss` for details.
    
    Args:
        log_probs: The logarithmized probabilities of the outputs.
        targets: Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
        input_lengths: Lengths of the inputs.
        target_lengths: Lengths of the targets.
        blank: Black label, default 0.
        reduction: Specifies the reduction to apply to the output:  ``'none'`` | ``'mean'`` | ``'sum'`` . Default ``'Mean'``.
        zero_infinity: Whether to zero infinite losses and the associated gradients. Default ``False``.
        
    Example:
        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        >>> import oneflow.nn.functional as F
        >>> log_probs = flow.tensor(
        ...     [
        ...         [[-1.1031, -0.7998, -1.5200], [-0.9808, -1.1363, -1.1908]],
        ...         [[-1.2258, -1.0665, -1.0153], [-1.1135, -1.2331, -0.9671]],
        ...         [[-1.3348, -0.6611, -1.5118], [-0.9823, -1.2355, -1.0941]],
        ...         [[-1.3850, -1.3273, -0.7247], [-0.8235, -1.4783, -1.0994]],
        ...         [[-0.9049, -0.8867, -1.6962], [-1.4938, -1.3630, -0.6547]],
        ...     ],
        ...     dtype=flow.float32,
        ...     requires_grad=True,
        ...     )
        >>> targets = flow.tensor([[1, 2, 2], [1, 2, 2]], dtype=flow.int32, device="cuda")
        >>> input_lengths = flow.tensor([5, 5], dtype=flow.int32)
        >>> target_lengths = flow.tensor([3, 3], dtype=flow.int32)
        >>> out = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor(1.1376, dtype=oneflow.float32, grad_fn=<scalar_mul_backward>)
        
    """
    max_target_length = 0
    if targets.ndim == 1:
        max_target_length = target_lengths.max().item()
    elif targets.ndim == 2:
        max_target_length = targets.shape[1]
    return flow._C.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        max_target_length,
        blank,
        zero_infinity,
        reduction,
    )
