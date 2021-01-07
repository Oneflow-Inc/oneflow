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
import oneflow_api
from typing import Optional


@oneflow_export("smooth_l1_loss")
def smooth_l1_loss(
    prediction: oneflow_api.BlobDesc,
    label: oneflow_api.BlobDesc,
    beta: float = 1.0,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator computes the smooth l1 loss. 

    The equation is: 

    .. math:: 

        & out = \frac{(\beta*x)^2}{2}, \left|x\right|<\frac{1}{{\beta}^2}

        & out = \left|x\right|-\frac{0.5}{{\beta}^2}, otherwise


    Args:
        prediction (oneflow_api.BlobDesc): The prediction Blob
        label (oneflow_api.BlobDesc): The label Blob
        beta (float, optional): The :math:`\beta` in the equation. Defaults to 1.0.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob 

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
