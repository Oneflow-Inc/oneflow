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
    oneflow.nn.functional.layer_norm,
    """nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor

    Applies Layer Normalization for last certain number of dimensions.

    See :class:`~oneflow.nn.LayerNorm` for details.

    """,
)
