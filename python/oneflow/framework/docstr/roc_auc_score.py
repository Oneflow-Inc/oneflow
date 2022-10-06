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
    oneflow.roc_auc_score,
    """
    oneflow.roc_auc_score(label, pred) -> Tensor

    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Note: Currently this implementation can only be used on CPU.

    Args:
        label (Tensor[N, 1]): True lable of the samples
        pred (Tensor[N, 1]): Predicted probability value to be true
        
    Returns:
        Tensor[1, ]: float32 tensor of auc score
       
    For example:

    .. code-block:: python


        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> label = flow.Tensor([0, 0, 1, 1])
        >>> pred = flow.Tensor([0.1, 0.4, 0.35, 0.8])     
          
        >>> score = flow.roc_auc_score(label, pred)
        >>> score
        tensor([0.7500], dtype=oneflow.float32)


    """,
)
