/*
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
*/
#include "oneflow/core/common/maybe.h"

namespace oneflow {

// align with pytorch: `c10/core/WrapDimMinimal.h`
static inline Maybe<int64_t> maybe_wrap_dim(int64_t dim, int64_t dim_post_expr,
                                            bool wrap_scalar = true) {
  if (dim_post_expr <= 0) {
    if (!wrap_scalar) {
      return Error::RuntimeError()
             << "dimension specified as " << dim << " but tensor has no dimensions";
    }
    dim_post_expr = 1;  // this will make range [-1, 0]
  }

  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  if (dim < min || dim > max) {
    return Error::IndexError() << "Dimension out of range (expected to be in range of [" << min
                               << ", " << max << "], but got " << dim << ")";
  }
  if (dim < 0) dim += dim_post_expr;
  return dim;
}
}  // namespace oneflow
