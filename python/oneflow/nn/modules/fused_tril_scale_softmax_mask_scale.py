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
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op


def fused_tril_scale_softmax_mask_scale_op(x, mask, diagonal=0, tril_fill_value=0, tril_scale_value=1, mask_scale_value=1):
    return flow.F.fused_tril_scale_softmax_mask_scale(x, mask, diagonal, tril_fill_value, tril_scale_value, mask_scale_value)


@register_tensor_op("fused_tril_scale_softmax_mask_scale")
def fused_tril_scale_softmax_mask_scale_op_tensor(x, mask, diagonal=0, tril_fill_value=0, tril_scale_value=1, mask_scale_value=1):
    return flow.F.fused_tril_scale_softmax_mask_scale(x, mask, diagonal, tril_fill_value, tril_scale_value, mask_scale_value)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
