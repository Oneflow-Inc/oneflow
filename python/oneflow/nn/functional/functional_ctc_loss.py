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
    reduction='mean', 
    zero_infinity=False
) -> Tensor:
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