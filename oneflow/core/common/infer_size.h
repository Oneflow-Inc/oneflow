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
#include <vector>
#include <sstream>
#include <string>
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

// align with pytorch: `aten\src\ATen\InferSize.h`

// Infers the size of a dim with size -1, if it exists. Also checks that new
// shape is compatible with the number of elements.
//
static inline Maybe<void> infer_size_impl(const Shape& shape, int64_t numel, DimVector& res) {
  int64_t newsize = 1;
  int64_t infer_dim = -1;
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (shape[dim] == -1) {
      CHECK_OR_RETURN(infer_dim == -1) << "only one dimension can be inferred";
      infer_dim = dim;
    } else {
      CHECK_OR_RETURN(shape[dim] >= 0) << "invalid shape dimension " << shape[dim];
      newsize *= shape[dim];
    }
  }
  CHECK_OR_RETURN(numel == newsize || (infer_dim >= 0 && newsize > 0 && numel % newsize == 0))
    << "shape '" << shape.ToString() << "' is invalid for input of size " << numel;
  if(infer_dim >= 0) {
    CHECK_OR_RETURN(newsize != 0) << "cannot reshape tensor of 0 elements into shape " << shape.ToString()
                                  << " because the unspecified dimension size -1 can be any value and is ambiguous";
    res[infer_dim] = numel / newsize;
  }
  return Maybe<void>::Ok();
}

static inline Maybe<DimVector> infer_size_dv(const Shape& shape, int64_t numel) {
  DimVector res = shape.dim_vec();
  infer_size_impl(shape, numel, res);
  return res;
}

}  // namespace oneflow
