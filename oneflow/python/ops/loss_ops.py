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
import oneflow.python.framework.remote_blob as remote_blob_util
from typing import Optional


@oneflow_export("smooth_l1_loss")
def smooth_l1_loss(
    prediction: remote_blob_util.BlobDef,
    label: remote_blob_util.BlobDef,
    beta: float = 1.0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""This operator computes the smooth l1 loss. 

    The equation is: 

    .. math:: 

        & out = \frac{(\beta*x)^2}{2}, \left|x\right|<\frac{1}{{\beta}^2}

        & out = \left|x\right|-\frac{0.5}{{\beta}^2}, otherwise


    Args:
        prediction (remote_blob_util.BlobDef): The prediction Blob
        label (remote_blob_util.BlobDef): The label Blob
        beta (float, optional): The :math:`\beta` in the equation. Defaults to 1.0.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The result Blob 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def smooth_l1_loss_Job(prediction: tp.Numpy.Placeholder((5, )),
                            label: tp.Numpy.Placeholder((5, ))
        ) -> tp.Numpy:
            return flow.smooth_l1_loss(prediction=prediction,
                                    label=label)


        prediction = np.array([0.1, 0.4, 0.3, 0.5, 0.9]).astype(np.float32)
        label = np.array([0.3, 0.9, 2.5, 0.4, 0.3]).astype(np.float32)
        out = smooth_l1_loss_Job(prediction, label)

        # out [0.02       0.12499999 1.7        0.005      0.17999998]

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SmoothL1Loss_")
        )
        .Op("smooth_l1_loss")
        .Input("prediction", [prediction])
        .Input("label", [label])
        .Output("loss")
    )
    op.Attr("beta", float(beta))
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("ctc_loss")
def ctc_loss(
    log_probs: remote_blob_util.BlobDef,  # Tensor of size (input_length, batch_size, number_of_classes)
    targets: remote_blob_util.BlobDef,  # Tensor of size (batch_size, max_target_length)
    input_lengths: remote_blob_util.BlobDef,  # Tuple or tensor of size (batch_size)
    target_lengths: remote_blob_util.BlobDef,  # Tuple or tensor of size (batch_size)
    blank: int = 0,  # blank label. Default 0.
    reduction: str = "mean",  # Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
    r"""Computes the CTC(Connectionist Temporal Classification) loss.
    This operator implements the CTC loss as presented in (Graves et al., 2006).


    Args:
        log_probs (remote_blob_util.BlobDef): A Blob of shape [input_length, batch_size, num_labels].
        targets (remote_blob_util.BlobDef): A Blob of shape [batch_size, max_target_length].
        input_lengths (remote_blob_util.BlobDef): A Blob of shape [batch_size].
        target_lengths (remote_blob_util.BlobDef): A Blob of shape [batch_size].
        blank (int, optional): Blank label. Defaults to 0.
        reduction (str, optional): The reduce type, it can be the one of "none", "mean", "sum". Defaults to "mean".
        zero_infinity (bool, optional):  Whether to zero infinite losses and the associated gradients. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        remote_blob_util.BlobDef: The result Blob.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        @flow.global_function()
        def ctc_loss_job(
            log_probs: tp.Numpy.Placeholder(shape=(5, 2, 3)),
            targets: tp.Numpy.Placeholder(shape=(2, 3), dtype=flow.int32),
            input_lengths: tp.Numpy.Placeholder(shape=(2,), dtype=flow.int32),
            target_lengths: tp.Numpy.Placeholder(shape=(2,), dtype=flow.int32),
        ) -> tp.Numpy:
            out = flow.ctc_loss(
                log_probs, targets, input_lengths, target_lengths, blank=0, reduction="none"
            )
            return out


        log_input = np.array(
            [
                [[-1.1031, -0.7998, -1.5200], [-0.9808, -1.1363, -1.1908]],
                [[-1.2258, -1.0665, -1.0153], [-1.1135, -1.2331, -0.9671]],
                [[-1.3348, -0.6611, -1.5118], [-0.9823, -1.2355, -1.0941]],
                [[-1.3850, -1.3273, -0.7247], [-0.8235, -1.4783, -1.0994]],
                [[-0.9049, -0.8867, -1.6962], [-1.4938, -1.3630, -0.6547]],
            ]
        ).astype(np.float32)
        target = np.array([[1, 2, 2], [1, 2, 2]]).astype("int64")
        input_lengths = np.array([5, 5]).astype("int64")
        target_lengths = np.array([3, 3]).astype("int64")
        loss = ctc_loss_job(log_input, target, input_lengths, target_lengths)

        # loss [3.918017 2.907672]

    """
    name = name if name is not None else id_util.UniqueStr("CTCLoss_")
    loss, alpha = (
        flow.user_op_builder(name)
        .Op("ctc_loss")
        .Input("log_probs", [log_probs])
        .Input("targets", [targets])
        .Input("input_lengths", [input_lengths])
        .Input("target_lengths", [target_lengths])
        .Output("loss")
        .Output("alpha")
        .Attr("blank", blank)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )

    if reduction == "mean":
        return flow.math.reduce_mean(
            flow.math.xdivy(
                loss,
                flow.cast(
                    flow.math.clip_by_value(target_lengths, min_value=1),
                    dtype=flow.float,
                ),
            ),
            name=name + "_reduce_mean",
        )
    elif reduction == "sum":
        return flow.math.reduce_sum(loss, name=name + "_reduce_sum")
    else:
        return loss
