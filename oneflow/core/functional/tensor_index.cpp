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
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace {

int64_t CountSpecifiedDims(const TensorIndex& index) {
  int64_t specified_ndims = 0;
  for (int i = 0; i < index.size(); ++i) {
    const auto& index_item = index.at(i);
    if (index_item.IsSlice() || index_item.IsInteger()) {
      specified_ndims++;
    } else if (index_item.IsTensor()) {
      const auto& tensor = index_item.tensor();
      if (IsMaskTensor(tensor)) {
        specified_ndims += tensor->ndim();
      } else {
        specified_ndims++;
      }
    }
  }
  return specified_ndims;
}

Maybe<TensorTuple> ExpandMaskIndex(const std::shared_ptr<Tensor>& index) {
  auto indices = std::make_shared<TensorTuple>();
  const auto& res = JUST(functional::ArgWhere(index, DType::Int64()));
  if (res->size() != 2) {
    return Error::RuntimeError() << "Argwhere should returns 2 tensors, but got " << res->size();
  }
  auto size_tensor = res->at(1);
  if (!size_tensor->is_eager()) {
    return Error::RuntimeError()
           << "Advanced indexing by boolean(mask) tensor only valid in eager mode.";
  }
  if (size_tensor->is_global()) {
    // TODO(): check size_tensor sbp is broadcast.
    size_tensor = JUST(functional::GlobalToLocal(size_tensor, /*copy=*/false));
  }
  int64_t size = 0;
  const auto& callback = [&](ep::Stream* stream,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    AutoMemcpy(stream, &size, eager_blob_object->dptr(), sizeof(size), memory::MakeHostMemCase(),
               eager_blob_object->mem_case());
  };
  JUST(SyncAccessTensorWithTimeOut(size_tensor, callback, "const"));

  for (int i = 0; i < index->ndim(); ++i) {
    auto item = JUST(functional::Slice((*res)[0], {0, i}, {size, i + 1}, {1, 1},
                                       /*enable_view_slice=*/false));
    item = JUST(functional::Reshape(item, {size}));
    indices->emplace_back(item);
  }
  return indices;
}

// NOTE: expand each non-empty indice to same shape.
Maybe<TensorTuple> ExpandIndices(const TensorTuple& indices) {
  std::shared_ptr<const Shape> expanded_shape;
  {
    bool first = true;
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
                << Error::RuntimeError() << "The size of tensor a (" << size
                << ") must match the size of tensor b (" << expanded_size
                << ") at non-singleton dimension " << i;
            sizes[j] = size == 1 ? expanded_size : size;
          }
        }
        expanded_shape.reset(new Shape(sizes));
      }
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

// NOTE(wyg):
// Judge whether all index dims are contiguous.
// e.g. [:, index0, index1, :] -> True
// [index0, :, index1] -> False
// [index0, index1, :] -> True
Maybe<bool> IsContinuousSubspace(const TensorTuple& indices) {
  int token = 0;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices.at(i) && !token) {
      token = 1;
    } else if (indices.at(i) && token) {
      if (token != 1) { return false; }
    } else if (token) {
      token += 1;
    }
  }
  return true;
}

// NOTE(wyg):
// Move indices subspace to be contiguous and ahead.
// e.g. [:, index0, index1] -> [index0, index1, :]
Maybe<std::vector<int>> TransposeFront(const std::shared_ptr<Tensor>& input,
                                       const TensorTuple& indices, std::shared_ptr<Tensor>* output,
                                       TensorTuple* valid_indices) {
  std::vector<int> permute;
  permute.reserve(input->ndim());
  for (int i = 0; i < input->ndim(); ++i) {
    if (i < indices.size() && indices.at(i)) {
      permute.emplace_back(i);
      valid_indices->emplace_back(indices.at(i));
    }
  }
  for (int i = 0; i < input->ndim(); ++i) {
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
  return permute;
}

Maybe<Tensor> AdjustSubspace(const std::shared_ptr<Tensor>& input, const TensorTuple& indices,
                             const int& index_ndim, bool reverse = false) {
  int index_subspace_pos = -1;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices.at(i)) {
      index_subspace_pos = i;
      break;
    }
  }
  if (index_subspace_pos <= 0) { return input; }
  int ndim = input->ndim();
  CHECK_LE_OR_RETURN(index_subspace_pos + index_ndim, ndim)
      << Error::IndexError()
      << "Failed to adjust subspace since the index is out of bounds for tensor dimension " << ndim;
  std::vector<int> permute;
  {
    permute.reserve(ndim);
    if (reverse) {
      for (int i = 0; i < index_ndim; ++i) { permute.emplace_back(index_subspace_pos + i); }
      for (int i = 0; i < index_subspace_pos; ++i) { permute.emplace_back(i); }
    } else {
      for (int i = 0; i < index_subspace_pos; ++i) { permute.emplace_back(i + index_ndim); }
      for (int i = 0; i < index_ndim; ++i) { permute.emplace_back(i); }
    }
    for (int i = permute.size(); i < ndim; ++i) { permute.emplace_back(i); }
  }
  return Transpose(input, permute);
}

Maybe<bool> HasFalseIndex(const TensorIndex& index) {
  return std::any_of(index.begin(), index.end(), [](const detail::IndexItem& item) {
    return item.IsBoolean() && !item.boolean();
  });
}

bool IsValidScalarTensorIndex(const std::shared_ptr<one::Tensor>& tensor) {
  if (!(tensor->dtype()->is_integer() || tensor->dtype() == DType::Bool())) { return false; }
  return tensor->shape()->NumAxes() == 0 && tensor->shape()->elem_cnt() == 1;
}

// Permute back for global tensor which transpose dims to front
Maybe<Tensor> PermuteBackForGlobalTensor(const std::shared_ptr<Tensor>& result,
                                         const std::vector<int>& permute) {
  CHECK_OR_RETURN(result->is_global());                // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(result->ndim(), permute.size());  // NOLINT(maybe-need-error-msg)
  std::vector<int> inv_permute(permute.size());
  for (int32_t i = 0; i < permute.size(); ++i) { inv_permute[permute[i]] = i; }

  bool not_permute = true;
  {
    for (int32_t i = 0; i < permute.size(); ++i) {
      if (inv_permute[i] != i) {
        not_permute = false;
        break;
      }
    }
  }
  if (!not_permute) {
    return Transpose(result, inv_permute);
  } else {
    return result;
  }
}

}  // namespace

bool IsMaskTensor(const std::shared_ptr<Tensor>& tensor) {
  return tensor->dtype() == DType::Int8() || tensor->dtype() == DType::UInt8()
         || tensor->dtype() == DType::Bool();
}

Maybe<void> PrepareSliceIndices(const TensorIndex& index, const Shape& shape,
                                std::vector<detail::Slice>* slice_indices,
                                TensorTuple* tensor_indices, std::vector<int64_t>* expand_dims,
                                std::vector<int64_t>* target_dims) {
  int64_t ndims = shape.NumAxes();
  int64_t specified_ndims = CountSpecifiedDims(index);
  CHECK_LE_OR_RETURN(specified_ndims, ndims)
      << Error::IndexError() << "Too many indices for tensor of dimension " << ndims;
  bool has_false_index = JUST(HasFalseIndex(index));
  bool has_expand_boolean_dim = false;
  int dim = 0;
  for (int i = 0; i < index.size(); ++i) {
    const auto& index_item = index.at(i);
    if (index_item.IsNone()) {
      expand_dims->emplace_back(dim);
      slice_indices->emplace_back(0, 1, 1);
      target_dims->emplace_back(1);
      continue;
    }
    if (index_item.IsBoolean()) {
      if (!has_expand_boolean_dim) {
        int boolean_index = !has_false_index;
        expand_dims->emplace_back(dim);
        slice_indices->emplace_back(0, boolean_index, 1);
        target_dims->emplace_back(boolean_index);
        has_expand_boolean_dim = true;
      }
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
    CHECK_LT_OR_RETURN(dim, ndims)
        << Error::IndexError() << "Invalid index for tensor of dimension " << ndims;
    if (index_item.IsSlice()) {
      const auto& slice = index_item.slice();
      CHECK_GT_OR_RETURN(slice.step(), 0)
          << Error::RuntimeError() << "Step must be greater than zero.";
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
      const auto& tensor = index_item.tensor();
      if (IsValidScalarTensorIndex(tensor) && !LazyMode::is_enabled()) {
        if (tensor->dtype()->is_integer() && tensor->dtype()->data_type() != DataType::kUInt8) {
          int64_t integer = JUST(GetItemInScalarTensor<int64_t>(tensor));
          if (integer < 0) { integer += shape.At(dim); }
          if (integer < 0 || integer >= shape.At(dim)) {
            return Error::IndexError()
                   << "Index " << index_item.integer() << " is out of bounds for dimension " << dim
                   << " with size " << shape.At(dim);
          }
          slice_indices->emplace_back(integer, integer + 1, 1);
          dim++;
        } else {
          bool boolean_index = JUST(GetItemInScalarTensor<bool>(tensor));
          if (!has_expand_boolean_dim) {
            expand_dims->emplace_back(dim);
            slice_indices->emplace_back(0, boolean_index, 1);
            target_dims->emplace_back(boolean_index);
            has_expand_boolean_dim = true;
          }
        }
      } else {
        auto indices = std::make_shared<TensorTuple>();
        if (tensor->dtype() == DType::Int8() || tensor->dtype() == DType::UInt8()
            || tensor->dtype() == DType::Bool()) {
          for (int j = 0; j < tensor->ndim(); ++j) {
            if (tensor->shape()->At(j) != shape.At(dim + j)) {
              return Error::IndexError()
                     << "The shape of the mask " << tensor->shape()->ToString() << " at index " << j
                     << " does not match the shape of the indexed tensor " << shape.ToString()
                     << " at index " << dim + j;
            }
          }
          indices = JUST(ExpandMaskIndex(tensor));
        } else {
          indices->emplace_back(tensor);
        }
        for (int j = 0; j < indices->size(); ++j) {
          slice_indices->emplace_back(0, shape.At(dim), 1);
          tensor_indices->resize(target_dims->size());
          tensor_indices->emplace_back(indices->at(j));
          target_dims->emplace_back(shape.At(dim));
          dim++;
        }
      }
    }
  }
  for (int i = dim; i < ndims; ++i) {
    slice_indices->emplace_back(0, shape.At(i), 1);
    target_dims->emplace_back(shape.At(i));
  }
  return Maybe<void>::Ok();
}

Maybe<std::vector<detail::Slice>> RemoveExpandDimSlice(
    const std::vector<detail::Slice>& expand_slices, const std::vector<int64_t>& expand_dims) {
  auto slices = std::make_shared<std::vector<detail::Slice>>();
  std::vector<int> mask(expand_slices.size(), 0);
  for (const auto& dim : expand_dims) {
    if (dim >= expand_slices.size()) {
      return Error::IndexError() << "Dimension " << dim << " is out of bounds for size "
                                 << expand_slices.size();
    }
    mask[dim] = 1;
  }
  for (int i = 0; i < expand_slices.size(); ++i) {
    if (!mask[i]) { slices->emplace_back(expand_slices.at(i)); }
  }
  return slices;
}

Maybe<Tensor> ApplyAdvancedIndexing(const std::shared_ptr<Tensor>& input,
                                    const TensorTuple& indices) {
  CHECK_GE_OR_RETURN(input->ndim(), indices.size())
      << Error::IndexError() << "Too many indices for tensor of dimension " << input->ndim();
  const auto& expanded_indices = JUST(ExpandIndices(indices));
  bool is_continuous_subspace = JUST(IsContinuousSubspace(indices));

  // Since the start dimension cannot be specified for `gather_nd`, so we should
  // transpose the input as long as the first index is null.
  std::shared_ptr<Tensor> transposed_input;
  TensorTuple valid_indices;
  JUST(TransposeFront(input, *expanded_indices, &transposed_input, &valid_indices));
  if (valid_indices.empty()) { return input; }
  int index_ndim = valid_indices.at(0)->ndim();
  std::shared_ptr<Tensor> packed_indices;
  if (valid_indices.size() == 1) {
    packed_indices = JUST(functional::Unsqueeze(valid_indices.at(0), -1));
  } else {
    packed_indices = JUST(Stack(valid_indices, 0));
    int packed_ndim = packed_indices->ndim();
    CHECK_GT_OR_RETURN(packed_ndim, 0)
        << Error::RuntimeError() << "Index array dimension should be greater than 0.";
    std::vector<int> permute(packed_ndim);
    permute[packed_ndim - 1] = 0;
    std::iota(permute.begin(), permute.end() - 1, 1);
    packed_indices = JUST(Transpose(packed_indices, permute))->contiguous();
  }

  CHECK_EQ_OR_RETURN(transposed_input->is_local(), packed_indices->is_local())
      << Error::RuntimeError() << "The input and indices must be both local or global.";

  auto result = JUST(GatherNd(transposed_input, packed_indices));

  int required_ndim = input->ndim() - valid_indices.size() + index_ndim;
  CHECK_EQ_OR_RETURN(result->ndim(), required_ndim)
      << Error::RuntimeError() << "The indexing result dimension is " << result->ndim()
      << ", but shoule be " << required_ndim;
  if (is_continuous_subspace) {
    result = JUST(AdjustSubspace(result, indices, index_ndim, /*reverse*/ false));
  }
  return result;
}

Maybe<Tensor> ApplyAdvancedIndexingUpdate(const std::shared_ptr<Tensor>& input,
                                          const TensorTuple& indices,
                                          const std::shared_ptr<Tensor>& value) {
  CHECK_GE_OR_RETURN(input->ndim(), indices.size())
      << Error::IndexError() << "Too many indices for tensor of dimension " << input->ndim();
  const auto& expanded_indices = JUST(ExpandIndices(indices));
  bool is_continuous_subspace = JUST(IsContinuousSubspace(indices));

  // Since the start dimension cannot be specified for `scatter_nd`, so we should
  // transpose the input as long as the first index is null.
  std::shared_ptr<Tensor> transposed_input;
  TensorTuple valid_indices;
  const auto& transposed_input_permute =
      JUST(TransposeFront(input, *expanded_indices, &transposed_input, &valid_indices));
  // NOTE: For local tensor, we make sure that transposed_input is a view of input.
  //       Therefore we need not transpose it back because we update the value in a same memory
  //       by tensor_scatter_nd_update operator.
  if (input->is_local()) {
    CHECK_EQ_OR_RETURN(JUST(transposed_input->tensor_storage()), JUST(input->tensor_storage()))
        << Error::RuntimeError()
        << "This setitem operator must enable view mechanism, please try to set "
           "ONEFLOW_DISABLE_VIEW=0";
  }

  if (valid_indices.empty()) {
    CHECK_EQ_OR_RETURN(value->nelement(), 0) << Error::IndexError() << "invalid indices";
    return input;
  }
  int index_ndim = valid_indices[0]->ndim();
  auto packed_indices = JUST(Stack(valid_indices, 0));
  {
    int packed_ndim = packed_indices->ndim();
    CHECK_GT_OR_RETURN(packed_ndim, 0)
        << Error::RuntimeError() << "Index array dimension should be greater than 0.";
    std::vector<int> permute(packed_ndim);
    permute[packed_ndim - 1] = 0;
    std::iota(permute.begin(), permute.end() - 1, 1);
    packed_indices = JUST(Transpose(packed_indices, permute))->contiguous();
  }

  CHECK_EQ_OR_RETURN(transposed_input->is_local(), packed_indices->is_local())
      << Error::RuntimeError() << "The input and indices must be both local or global.";

  Shape expand_shape;
  {
    if (is_continuous_subspace) {
      bool index_subspace_begin = true;
      for (int i = 0; i < indices.size(); ++i) {
        // if the index is the first not-null index
        if (indices[i]) {
          if (!index_subspace_begin) { continue; }
          for (int j = 0; j < index_ndim; ++j) {
            expand_shape.emplace_back(valid_indices[0]->shape()->At(j));
          }
          index_subspace_begin = false;
        } else {
          expand_shape.emplace_back(input->shape()->At(i));
        }
      }
    } else {
      expand_shape = *(valid_indices[0]->shape());
      for (int i = 0; i < indices.size(); ++i) {
        if (!indices[i]) { expand_shape.emplace_back(input->shape()->At(i)); }
      }
    }
    for (int i = indices.size(); i < input->ndim(); ++i) {
      expand_shape.emplace_back(input->shape()->At(i));
    }
  }
  std::shared_ptr<Tensor> expand_value = JUST(Expand(value, expand_shape));

  // reverse adjust value if index subspace is continuous but transposed since the start
  // dimension cannot be specified for `scatter_nd`
  if (is_continuous_subspace) {
    expand_value = JUST(AdjustSubspace(expand_value, indices, index_ndim, /*reverse*/ true));
  }
  JUST(TensorScatterNdUpdate(transposed_input, packed_indices, expand_value, /*inplace=*/true));
  // Global tensor is not support view, so we should permute back and copy to origin input if need
  if (transposed_input->is_global()) {
    return PermuteBackForGlobalTensor(transposed_input, *transposed_input_permute);
  }
  return transposed_input;
}

Maybe<Tensor> ApplySelectIndexing(const std::shared_ptr<one::Tensor>& input,
                                  const TensorIndex& tensor_index) {
  const int32_t index = tensor_index[0].integer();
  const int32_t ndim = input->ndim();
  CHECK_OR_RETURN(ndim > 0) << Error::RuntimeError()
                            << "select() cannot be applied to a 0-dim tensor.";
  const int32_t pos_dim = 0;
  auto size = input->dim(pos_dim);
  CHECK_OR_RETURN(index >= -size && index < size)
      << Error::IndexError() << "Index out of range (expected to be in range of [" << -size << ","
      << size - 1 << "], but got " << index << ")";
  int32_t pos_index = index >= 0 ? index : index + size;
  std::vector<int64_t> sizes(input->shape()->dim_vec().begin() + 1,
                             input->shape()->dim_vec().end());
  const auto& stride = *JUST(input->stride());
  const int64_t storage_offset = JUST(input->storage_offset()) + pos_index * stride[pos_dim];
  std::vector<int64_t> strides(stride.begin() + 1, stride.end());
  return functional::AsStrided(input, sizes, strides, storage_offset);
}

Maybe<void> UnifyInputAndIndicesOnDevice(const std::shared_ptr<Tensor>& x,
                                         TensorTuple& tensor_indices) {
  if (x->is_local()) {
    const auto x_device = JUST(x->device());
    for (int64_t i = 0; i < tensor_indices.size(); ++i) {
      const auto tensor_index = tensor_indices[i];
      if (tensor_index == nullptr) { continue; }
      if (tensor_index->is_global()) { return Maybe<void>::Ok(); }
      const auto tensor_index_device = JUST(tensor_index->device());
      if ((tensor_index_device->type() != x_device->type())
          || (tensor_index_device->device_id() != x_device->device_id())) {
        tensor_indices[i] =
            JUST(Copy(tensor_index, x_device->type(), x_device->device_id(), /*pin_memory=*/false));
      }
    }
  } else {
    // global tensor
    const auto& placement = JUST(x->parallel_desc());
    const auto& broadcast_sbp = JUST(MakeBroadcastSbpParallel());
    int n = JUST(x->nd_sbp())->sbp_parallel_size();
    std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
    for (int64_t i = 0; i < tensor_indices.size(); ++i) {
      const auto tensor_index = tensor_indices[i];
      if (tensor_index == nullptr) { continue; }
      if (tensor_index->is_local()) {
        // NOTE: LocalToGlobal should be called in eager mode
        LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
        tensor_indices[i] = JUST(ToGlobal(tensor_index, placement,
                                          std::vector<Symbol<SbpParallel>>(n, broadcast_sbp),
                                          grad_sbp_tuple, /*check_meta=*/false, /*copy=*/false));
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
