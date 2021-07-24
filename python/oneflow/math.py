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
from oneflow.ops.math_binary_elementwise_ops import atan2, floordiv, pow, xdivy, xlogy
from oneflow.ops.reduce_mean import reduce_mean
from oneflow.ops.reduce_ops import (
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
from oneflow.ops.two_stage_reduce import (
    api_two_stage_reduce_max as two_stage_reduce_max,
)
from oneflow.ops.two_stage_reduce import (
    api_two_stage_reduce_min as two_stage_reduce_min,
)
