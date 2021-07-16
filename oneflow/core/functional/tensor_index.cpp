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

Maybe<TensorIndex> RegularTensorIndex(const TensorIndex& index, const Shape& shape) {
  int64_t specified_ndims = CountSpecifiedDims(index);
  int64_t ndims = shape.NumAxes();
  CHECK_LE_OR_RETURN(specified_ndims, ndims)
      << "Too many indices for tensor of dimension " << ndims;

  auto regular_index = std::make_shared<TensorIndex>();
  int64_t dim = 0;
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
      regular_index->emplace_back(detail::IndexItem(start, end, step));
      dim++;
    } else if (index_item.IsInteger()) {
      CHECK_LT_OR_RETURN(dim, ndims) << "Invalid index for tensor of dimension " << ndims;
      int64_t integer = index_item.integer();
      if (integer < 0) { integer += shape.At(dim); }
      CHECK_OR_RETURN(integer >= 0 && integer < shape.At(dim)) << Error::ValueError(
          std::string("Index ") + std::to_string(index_item.integer())
          + std::string(" is out of bounds for dimension ") + std::to_string(dim)
          + std::string(" with size ") + std::to_string(shape.At(dim)));
      regular_index->emplace_back(detail::IndexItem(integer));
      dim++;
    } else if (index_item.IsEllipsis()) {
      int64_t unspecified_ndims = ndims - specified_ndims;
      unspecified_ndims = std::min(ndims - dim, unspecified_ndims);
      for (int j = 0; j < unspecified_ndims; ++j) {
        regular_index->emplace_back(detail::IndexItem(0, shape.At(dim + j), 1));
      }
      dim += unspecified_ndims;
    } else {
      // None or Boolean.
      if (index_item.IsBoolean()) {
        CHECK_OR_RETURN(index_item.boolean()) << "Index false is not supported.";
      }
      regular_index->emplace_back(index_item);
    }
  }
  for (int i = dim; i < ndims; ++i) {
    regular_index->emplace_back(detail::IndexItem(0, shape.At(i), 1));
  }
  return regular_index;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
