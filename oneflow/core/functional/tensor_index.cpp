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

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace one {
namespace functional {

int64_t CountSpecifiedDims(const TensorIndex& index) {
  int64_t specified_ndims = 0;
  for (int i = 0; i < index.size(); ++i) {
    const auto& index_item = index.at(i);
    if (index_item.IsSlice() || index_item.IsInteger() || index_item.IsTensor()) {
      specified_ndims++;
    }
  }
  return specified_ndims;
}

Maybe<void> PrepareSliceIndices(const TensorIndex& index, const Shape& shape,
                                std::vector<detail::Slice>* slice_indices,
                                TensorTuple* tensor_indices, std::vector<int64_t>* target_dims) {
  int64_t ndims = shape.NumAxes();
  int64_t specified_ndims = CountSpecifiedDims(index);
  CHECK_LE_OR_RETURN(specified_ndims, ndims)
      << "Too many indices for tensor of dimension " << ndims;
  int dim = 0;
  for (int i = 0; i < index.size(); ++i) {
    const auto& index_item = index.at(i);
    if (index_item.IsNone()) {
      target_dims->emplace_back(1);
      continue;
    }
    if (index_item.IsBoolean()) {
      target_dims->emplace_back(index_item.boolean());
      continue;
    }
    if (index_item.IsEllipsis()) {
      int64_t unspecified_ndims = ndims - specified_ndims;
      unspecified_ndims = std::min(ndims - dim, unspecified_ndims);
      for (int j = 0; j < unspecified_ndims; ++j) {
        slice_indices->emplace_back(0, shape.At(dim + j), 1);
        target_dims->emplace_back(shape.At(dim + j));
      }
      dim += unspecified_ndims;
      continue;
    }
    CHECK_LT_OR_RETURN(dim, ndims) << "Invalid index for tensor of dimension " << ndims;
    if (index_item.IsSlice()) {
      const auto& slice = index_item.slice();
      CHECK_GT_OR_RETURN(slice.step(), 0) << "Step must be greater than zero.";
      int64_t step = std::min(slice.step(), shape.At(dim));
      int64_t end = std::min(slice.end(), shape.At(dim));
      int64_t start = std::min(slice.start(), shape.At(dim));
      if (start < 0) { start += shape.At(dim); }
      if (start < 0) { start = 0; }
      if (end < 0) { end += shape.At(dim); }
      if (end < start) { end = start; }
      if (start == end) { step = 1; }
      slice_indices->emplace_back(start, end, step);
      int64_t length = start == end ? 0 : (end - start + step - 1) / step;
      target_dims->emplace_back(length);
      dim++;
    } else if (index_item.IsInteger()) {
      int64_t integer = index_item.integer();
      if (integer < 0) { integer += shape.At(dim); }
      if (integer < 0 || integer >= shape.At(dim)) {
        return Error::IndexError()
               << "Index " << index_item.integer() << " is out of bounds for dimension " << dim
               << " with size " << shape.At(dim);
      }
      slice_indices->emplace_back(integer, integer + 1, 1);
      dim++;
    } else if (index_item.IsTensor()) {
      slice_indices->emplace_back(0, shape.At(dim), 1);
      tensor_indices->resize(target_dims->size());
      tensor_indices->emplace_back(index_item.tensor());
      target_dims->emplace_back(shape.At(dim));
      dim++;
    }
  }
  for (int i = dim; i < ndims; ++i) {
    slice_indices->emplace_back(0, shape.At(i), 1);
    target_dims->emplace_back(shape.At(i));
  }
  return Maybe<void>::Ok();
}

namespace {

Maybe<TensorTuple> ExpandIndices(const TensorTuple& indices) {
  bool first = true;
  std::shared_ptr<const Shape> expanded_shape;
  for (int i = 0; i < indices.size(); ++i) {
    if (!indices.at(i)) { continue; }
    if (first) {
      expanded_shape = indices.at(i)->shape();
      first = false;
    } else {
      const auto& shape = indices.at(i)->shape();
      int ndims = std::max(shape->NumAxes(), expanded_shape->NumAxes());
      DimVector sizes(ndims);
      for (int j = ndims - 1; j >= 0; --j) {
        int dim = j - (ndims - shape->NumAxes());
        int expanded_dim = j - (ndims - expanded_shape->NumAxes());
        if (dim < 0) {
          sizes[j] = expanded_shape->At(expanded_dim);
        } else if (expanded_dim < 0) {
          sizes[j] = shape->At(dim);
        } else {
          int size = shape->At(dim);
          int expanded_size = expanded_shape->At(expanded_dim);
          CHECK_OR_RETURN(size == expanded_size || size == 1 || expanded_size == 1)
              << "Cannot broadcast advanced index to size " << std::max(size, expanded_size)
              << " at dimension " << j << " since the size of another index is not 1.";
          sizes[j] = std::max(size, expanded_size);
        }
      }
      expanded_shape.reset(new Shape(sizes));
    }
  }
  auto expanded_indices = std::make_shared<TensorTuple>(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    if (!indices.at(i)) { continue; }
    if (*(indices.at(i)->shape()) != *expanded_shape) {
      expanded_indices->at(i) = JUST(Expand(indices.at(i), *expanded_shape));
    } else {
      expanded_indices->at(i) = indices.at(i);
    }
  }
  return expanded_indices;
}

Maybe<bool> IsContinuosSubspace(const TensorTuple& indices) {
  int token = 0;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices.at(i) && !token) {
      token = 1;
    } else if (indices.at(i) && token) {
      if (token != 1) { return false; }
    } else if (!token) {
      token += 1;
    }
  }
  return true;
}

Maybe<void> TransposeFront(const std::shared_ptr<Tensor>& input, const TensorTuple& indices,
                           std::shared_ptr<Tensor>* output, TensorTuple* valid_indices) {
  std::vector<int> permute;
  permute.reserve(input->shape()->NumAxes());
  for (int i = 0; i < input->shape()->NumAxes(); ++i) {
    if (i < indices.size() && indices.at(i)) {
      permute.emplace_back(i);
      valid_indices->emplace_back(indices.at(i));
    }
  }
  for (int i = 0; i < input->shape()->NumAxes(); ++i) {
    if (i >= indices.size() || !indices.at(i)) { permute.emplace_back(i); }
  }
  bool need_transpose = [&]() {
    for (int i = 0; i < permute.size(); ++i) {
      if (permute.at(i) != i) { return true; }
    }
    return false;
  }();
  if (need_transpose) {
    *output = JUST(Transpose(input, permute));
  } else {
    *output = input;
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> AdjustSubspace(const std::shared_ptr<Tensor>& input, const TensorTuple& indices,
                             const int& index_ndim) {
  int index_subspace_pos = -1;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices.at(i)) {
      index_subspace_pos = i;
      break;
    }
  }
  if (index_subspace_pos <= 0) { return input; }
  int ndim = input->shape()->NumAxes();
  CHECK_LE_OR_RETURN(index_subspace_pos + index_ndim, ndim)
      << "Failed to adjust subspace since the index is out of bounds for tensor dimension " << ndim;
  std::vector<int> permute;
  permute.reserve(ndim);
  for (int i = 0; i < index_subspace_pos; ++i) { permute.emplace_back(i + index_ndim); }
  for (int i = 0; i < index_ndim; ++i) { permute.emplace_back(i); }
  for (int i = permute.size(); i < ndim; ++i) { permute.emplace_back(i); }
  return Transpose(input, permute);
}

}  // namespace

Maybe<Tensor> ApplyAdvancedIndexing(const std::shared_ptr<Tensor>& input,
                                    const TensorTuple& indices) {
  CHECK_GE_OR_RETURN(input->shape()->NumAxes(), indices.size())
      << "Too many indices for tensor of dimension " << input->shape()->NumAxes();
  const auto& expanded_indices = JUST(ExpandIndices(indices));
  bool is_continuos_subspace = JUST(IsContinuosSubspace(indices));

  // Since the start dimension cannot be specified for `gather_nd`, so we should
  // transpose the input as long as the first indice is null.
  std::shared_ptr<Tensor> transposed_input;
  TensorTuple valid_indices;
  JUST(TransposeFront(input, *expanded_indices, &transposed_input, &valid_indices));
  if (valid_indices.empty()) { return input; }
  int index_ndim = valid_indices.at(0)->shape()->NumAxes();
  std::shared_ptr<Tensor> packed_indices;
  if (valid_indices.size() > 1) {
    packed_indices = JUST(Stack(valid_indices, 0));
  } else {
    packed_indices = JUST(ExpandDims(valid_indices.at(0), 0));
  }
  int packed_ndim = packed_indices->shape()->NumAxes();
  CHECK_GT_OR_RETURN(packed_ndim, 0) << "Index array dimension should be greater than 0.";
  std::vector<int> permute(packed_ndim);
  permute[packed_ndim - 1] = 0;
  std::iota(permute.begin(), permute.end() - 1, 1);
  packed_indices = JUST(Transpose(packed_indices, permute));

  if (transposed_input->is_consistent()) {
    const auto& placement = JUST(transposed_input->parallel_desc());
    const auto& broadcast_sbp = JUST(MakeBroadcastSbpParallel());
    packed_indices =
        JUST(ToConsistent(packed_indices, placement, {broadcast_sbp}, GetNoneSbpList()));
  }
  Symbol<Device> device = JUST(transposed_input->device());
  if (JUST(packed_indices->device()) != device) {
    packed_indices = JUST(Copy(packed_indices, device->type(), device->device_id()));
  }
  auto result = JUST(GatherNd(transposed_input, packed_indices));

  int required_ndim = input->shape()->NumAxes() - valid_indices.size() + index_ndim;
  CHECK_EQ_OR_RETURN(result->shape()->NumAxes(), required_ndim)
      << "The indexing result dimension is " << result->shape()->NumAxes() << ", but shoule be "
      << required_ndim;
  if (is_continuos_subspace) { result = JUST(AdjustSubspace(result, indices, index_ndim)); }
  return result;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
