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
from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow._oneflow_internal
from typing import Optional, Tuple


@oneflow_export("nn.ctc_greedy_decoder")
def ctc_greedy_decoder(
    log_probs: oneflow._oneflow_internal.BlobDesc,
    input_lengths: oneflow._oneflow_internal.BlobDesc,
    merge_repeated: bool = True,
    name: Optional[str] = None,
) -> Tuple[oneflow._oneflow_internal.BlobDesc, oneflow._oneflow_internal.BlobDesc]:
    r"""Performs greedy decoding on the logits given in input (best path).

    Args:
        log_probs (oneflow._oneflow_internal.BlobDesc): A Blob of shape [input_length, batch_size, num_labels]. The logarithmized probabilities of the outputs (e.g. obtained with flow.nn.logsoftmax()).
        input_lengths (oneflow._oneflow_internal.BlobDesc): A Blob of shape [batch_size]. It represent the lengths of the inputs. And the lengths are specified for each sequence to achieve masking under the assumption that sequences are padded to equal lengths.
        merge_repeated (bool, optional): If merge_repeated is True, merge repeated classes in output. This means that if consecutive logits' maximum indices are the same, only the first of these is emitted. Defaults to True.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        decoded(oneflow._oneflow_internal.BlobDesc): A Blob of shape [batch_size, input_length], The decoded outputs.
        neg_sum_logits(oneflow._oneflow_internal.BlobDesc): A float matrix (batch_size x 1) containing, for the sequence found, the negative of the sum of the greatest logit at each timeframe.

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np
        from typing import Tuple


        @flow.global_function()
        def ctc_greedy_decoder_job(
            log_probs: tp.Numpy.Placeholder(shape=(4, 2, 5)),
            input_lengths: tp.Numpy.Placeholder(shape=(2,), dtype=flow.int64),
        ) -> Tuple[tp.Numpy, tp.Numpy]:
            decoded, neg_sum_logits = flow.nn.ctc_greedy_decoder(
                log_probs, input_lengths, merge_repeated=True
            )
            return decoded, neg_sum_logits


        log_probs = np.array(
            [
                [[-1.54, -1.20, -1.95, -1.65, -1.81], [-1.84, -1.74, -1.58, -1.55, -1.12]],
                [[-1.68, -1.48, -1.89, -1.30, -2.07], [-1.13, -1.45, -1.24, -1.61, -1.66]],
                [[-1.56, -1.40, -2.83, -1.67, -1.48], [-1.20, -2.01, -2.05, -1.95, -1.24]],
                [[-2.09, -1.76, -1.36, -1.67, -1.45], [-1.85, -1.48, -1.34, -2.16, -1.55]],
            ]
        ).astype(np.float32)
        input_lengths = np.array([4, 4])
        decoded, neg_sum_logits = ctc_greedy_decoder_job(log_probs, input_lengths)

        # decoded [[1 3 1 2] [0 2 0 0]]
        # neg_sum_logits [[5.26] [4.79]]


    """
    name = name if name is not None else id_util.UniqueStr("CTCGreedyDecode_")
    decoded, neg_sum_logits = (
        flow.user_op_builder(name)
        .Op("ctc_greedy_decoder")
        .Input("log_probs", [log_probs])
        .Input("input_lengths", [input_lengths])
        .Output("decoded")
        .Output("neg_sum_logits")
        .Attr("merge_repeated", merge_repeated)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return decoded, neg_sum_logits


@oneflow_export("nn.ctc_beam_search_decoder")
def ctc_beam_search_decoder(
    log_probs: oneflow._oneflow_internal.BlobDesc,
    input_lengths: oneflow._oneflow_internal.BlobDesc,
    beam_width: int,
    top_paths: int,
    name: Optional[str] = None,
) -> Tuple[oneflow._oneflow_internal.BlobDesc, oneflow._oneflow_internal.BlobDesc]:
    r"""Performs beam search decoding on the logits given in input.

    Args:
        log_probs (oneflow._oneflow_internal.BlobDesc): A Blob of shape [input_length, batch_size, num_labels]. The logarithmized probabilities of the outputs (e.g. obtained with flow.nn.logsoftmax()).
        input_lengths (oneflow._oneflow_internal.BlobDesc): A Blob of shape [batch_size]. It represent the lengths of the inputs. And the lengths are specified for each sequence to achieve masking under the assumption that sequences are padded to equal lengths.
        beam_width (int, optional): An int scalar >= 0 (beam search beam width). 
        top_paths (int, optional): An int scalar >= 0, <= beam_width (controls output size).
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        decoded(oneflow._oneflow_internal.BlobDesc): A Blob of shape [batch_size, input_length], The decoded outputs.
        log_probability(oneflow._oneflow_internal.BlobDesc): A float matrix [batch_size, top_paths] containing sequence log-probabilities.

    For example:

    .. code-block:: python
    


    """
    name = name if name is not None else id_util.UniqueStr("CTCBeamSearchDecode_")
    decoded, neg_sum_logits = (
        flow.user_op_builder(name)
        .Op("ctc_beam_search_decoder")
        .Input("log_probs", [log_probs])
        .Input("input_lengths", [input_lengths])
        .Output("decoded")
        .Output("log_probability")
        .Attr("beam_width", beam_width)
        .Attr("top_paths", top_paths)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return decoded, neg_sum_logits
