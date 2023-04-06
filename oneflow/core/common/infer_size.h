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
  auto infer_dim = Optional<int64_t>();
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (shape[dim] == -1) {
      CHECK_OR_RETURN(infer_dim.has_value()) << "only one dimension can be inferred";
      infer_dim = dim;
    } else if (shape[dim] >= 0) {
      newsize *= shape[dim];
    } else {
      CHECK_OR_RETURN(false) << "invalid shape dimension " << shape[dim];
    }
  }

  if (numel == newsize || (infer_dim.has_value() && newsize > 0 && numel % newsize == 0)) {
    if (infer_dim.has_value()) {
      // We have a degree of freedom here to select the dimension size; follow
      // NumPy semantics and just bail.  However, a nice error message is needed
      // because users often use `view` as a way to flatten & unflatten
      // dimensions and will otherwise be confused why
      //   empty_tensor.view( 0, 0)
      // works yet
      //   empty_tensor.view(-1, 0)
      // doesn't.
      CHECK_OR_RETURN(newsize == 0) << "cannot reshape tensor of 0 elements into shape " << shape.ToString()
                                    << " because the unspecified dimension size -1 can be any value and is ambiguous";
      int64_t infer_dim_ = JUST(infer_dim);
      res[infer_dim_] = numel / newsize;
    }
    return Maybe<void>::Ok();
  }

  CHECK_OR_RETURN(false) << "shape '" << shape.ToString() << "' is invalid for input of size " << numel;
}

static inline Maybe<DimVector> infer_size_dv(const Shape& shape, int64_t numel) {
  DimVector res = shape.dim_vec();
  infer_size_impl(shape, numel, res);
  return res;
}

}  // namespace oneflow
