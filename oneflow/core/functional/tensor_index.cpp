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
#include "oneflow/core/functional/tensor_index.h"

namespace oneflow {
namespace one {
namespace functional {

int64_t CountSpecifiedDims(const TensorIndex& index) {
  int64_t specified_ndims = 0;
  for (int i = 0; i < index.size(); ++i) {
    const auto& index_item = index.at(i);
    if (index_item.IsSlice() || index_item.IsInteger()) { specified_ndims++; }
  }
  return specified_ndims;
}

Maybe<void> PrepareSliceIndices(const TensorIndex& index, const Shape& shape,
                                std::vector<detail::Slice>* slice_indices,
                                std::vector<std::shared_ptr<Tensor>>* tensor_indices,
                                std::vector<int64_t>* target_dims) {
  int64_t ndims = shape.NumAxes();
  int64_t specified_ndims = CountSpecifiedDims(index);
  CHECK_LE_OR_RETURN(specified_ndims, ndims)
      << "Too many indices for tensor of dimension " << ndims;
  int dim = 0;
  for (int i = 0; i < index.size(); ++i) {
    const auto& index_item = index.at(i);
    if (index_item.IsSlice()) {
      CHECK_LT_OR_RETURN(dim, ndims) << "Invalid index for tensor of dimension " << ndims;
      CHECK_GT_OR_RETURN(shape.At(dim), 0) << "Slice cannot be applied to a 0-dim tensor.";
      const auto& slice = index_item.slice();
      int64_t step = std::min(slice.step(), shape.At(dim));
      CHECK_GT_OR_RETURN(step, 0) << "Step must be greater than zero.";
      int64_t end = std::min(slice.end(), shape.At(dim));
      int64_t start = std::min(slice.start(), shape.At(dim));
      if (start < 0) { start += shape.At(dim); }
      if (start < 0) { start = 0; }
      if (end < 0) { end += shape.At(dim); }
      if (end < start) { end = start; }
      slice_indices->emplace_back(start, end, step);
      int64_t length = (end - start + step - 1) / step;
      target_dims->emplace_back(length);
      dim++;
    } else if (index_item.IsInteger()) {
      CHECK_LT_OR_RETURN(dim, ndims) << "Invalid index for tensor of dimension " << ndims;
      int64_t integer = index_item.integer();
      if (integer < 0) { integer += shape.At(dim); }
      if (integer < 0 && integer >= shape.At(dim)) {
        return Error::ValueError() << "Index " << index_item.integer() << " is out of bounds for dimension " << dim
            << " with size " << shape.At(dim);
      }
      slice_indices->emplace_back(integer, integer + 1, 1);
      dim++;
    } else if (index_item.IsEllipsis()) {
      int64_t unspecified_ndims = ndims - specified_ndims;
      unspecified_ndims = std::min(ndims - dim, unspecified_ndims);
      for (int j = 0; j < unspecified_ndims; ++j) {
        slice_indices->emplace_back(0, shape.At(dim + j), 1);
        target_dims->emplace_back(shape.At(dim + j));
      }
      dim += unspecified_ndims;
    } else if (index_item.IsNone()) {
      target_dims->emplace_back(1);
    } else if (index_item.IsBoolean()) {
      target_dims->emplace_back(index_item.boolean());
    }
  }
  for (int i = dim; i < ndims; ++i) {
    slice_indices->emplace_back(0, shape.At(i), 1);
    target_dims->emplace_back(shape.At(i));
  }
  return Maybe<void>::Ok();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
