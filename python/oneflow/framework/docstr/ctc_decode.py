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
    oneflow._C.ctc_greedy_decoder,
    """
    ctc_greedy_decoder(log_probs: Tensor, input_lengths: Tensor, merge_repeated: bool=True) -> Tensor

    Performs greedy decoding on the logits given in input (best path).

    Args:
        log_probs(oneflow.Tensor): A Tensor of shape [input_length, batch_size, num_labels]. The logarithmized probabilities of the outputs (e.g. obtained with flow.nn.logsoftmax()).
        input_lengths(oneflow.Tensor): A Tensor of shape [batch_size]. It represent the lengths of the inputs. And the lengths are specified for each sequence to achieve masking under the assumption that sequences are padded to equal lengths.
        merge_repeated (bool, optional): If merge_repeated is True, merge repeated classes in output. This means that if consecutive logits' maximum indices are the same, only the first of these is emitted. Defaults to True.

    Returns:
        decoded(oneflow.Tensor): A Tensor of shape [batch_size, input_length], The decoded outputs.
        neg_sum_logits(oneflow.Tensor): A float matrix (batch_size x 1) containing, for the sequence found, the negative of the sum of the greatest logit at each timeframe.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> log_probs = flow.tensor(
        ...     [
        ...         [[-1.54, -1.20, -1.95, -1.65, -1.81], [-1.84, -1.74, -1.58, -1.55, -1.12]],
        ...         [[-1.68, -1.48, -1.89, -1.30, -2.07], [-1.13, -1.45, -1.24, -1.61, -1.66]],
        ...         [[-1.56, -1.40, -2.83, -1.67, -1.48], [-1.20, -2.01, -2.05, -1.95, -1.24]],
        ...         [[-2.09, -1.76, -1.36, -1.67, -1.45], [-1.85, -1.48, -1.34, -2.16, -1.55]],
        ...     ]
        ... )
        >>> input_lengths = flow.tensor([4, 4])
        >>> decoded, neg_sum_logits = flow.nn.functional.ctc_greedy_decoder(log_probs, input_lengths)
        >>> decoded
        tensor([[1, 3, 1, 2],
                [0, 2, 0, 0]], dtype=oneflow.int64)
        >>> neg_sum_logits
        tensor([[5.2600],
                [4.7900]], dtype=oneflow.float32)

    """,
)
