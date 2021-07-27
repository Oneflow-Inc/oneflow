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
from oneflow.compatible.single_client.ops.math_binary_elementwise_ops import (
    atan2,
    floordiv,
    pow,
    xdivy,
    xlogy,
)
from oneflow.compatible.single_client.ops.math_ops import (
    add,
    add_n,
    argmax,
    broadcast_to_compatible_with,
    clip_by_value,
    divide,
)
from oneflow.compatible.single_client.ops.math_ops import (
    elem_cnt as reduced_shape_elem_cnt,
)
from oneflow.compatible.single_client.ops.math_ops import equal
from oneflow.compatible.single_client.ops.math_ops import floor_mod as mod
from oneflow.compatible.single_client.ops.math_ops import (
    fused_scale_tril,
    fused_scale_tril_softmax_dropout,
    gelu,
    gelu_grad,
    greater,
    greater_equal,
    in_top_k,
    l2_normalize,
    less,
    less_equal,
    logical_and,
    maximum,
    minimum,
    multiply,
    not_equal,
    polyval,
    relu,
    sigmoid,
    sigmoid_grad,
    squared_difference,
    subtract,
    top_k,
    tril,
    unsorted_batch_segment_sum,
    unsorted_segment_sum,
    unsorted_segment_sum_like,
)
from oneflow.compatible.single_client.ops.math_unary_elementwise_ops import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    ceil,
    cos,
    cosh,
    erf,
    erfc,
    exp,
    expm1,
    floor,
    lgamma,
    log,
    log1p,
    log_sigmoid,
    negative,
    reciprocal,
    reciprocal_no_nan,
    rint,
    round,
    rsqrt,
    sigmoid_v2,
    sign,
    sin,
    sinh,
    softplus,
    sqrt,
    square,
    tan,
    tanh,
    tanh_v2,
)
from oneflow.compatible.single_client.ops.reduce_mean import reduce_mean
from oneflow.compatible.single_client.ops.reduce_ops import (
    reduce_all,
    reduce_any,
    reduce_euclidean_norm,
    reduce_logsumexp,
    reduce_max,
    reduce_min,
    reduce_prod,
    reduce_std,
    reduce_sum,
    reduce_variance,
)
from oneflow.compatible.single_client.ops.two_stage_reduce import (
    api_two_stage_reduce_max as two_stage_reduce_max,
)
from oneflow.compatible.single_client.ops.two_stage_reduce import (
    api_two_stage_reduce_min as two_stage_reduce_min,
)
