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
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/common/wrap_dim_utils.h"

namespace oneflow {
namespace one {
namespace view {

// NOTE: use env variable 'ONEFLOW_DISABLE_VIEW' control use view mechanism or not
// If  set true, then do not use view mechanism(and view ops)
bool IsEnvViewDisabled() {
  static const bool env_view_disabled = ParseBooleanFromEnv("ONEFLOW_DISABLE_VIEW", false);
  return env_view_disabled;
}

bool IsViewApplicable(const std::shared_ptr<Tensor>& input) {
  if (IsEnvViewDisabled()) { return false; }
  // NOTE: only eager local tensor support view for now
  // elem_cnt() >= 1  used to excluding 0 shape tensor
  if (input->is_local() && !(LazyMode::is_enabled()) && input->shape()->elem_cnt() >= 1) {
    return true;
  }
  return false;
}

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        int64_t storage_offset) {
  /**
   * This function provides basic view capabilities which
   * accept input tensor with target shape, and return viewed tensor.
   *
   * The viewed tensor shared memory with input tensor, and both of
   * them are memory contiguous, but has different shapes/strides.
   */
  Stride target_stride(target_shape);
  return BasicView(input, target_shape, target_stride, storage_offset);
}

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const Stride& target_stride, int64_t storage_offset) {
  // TODO(): Check shape compatible.
  auto device = JUST(input->device());
  auto tensor_meta = std::make_shared<LocalTensorMeta>(
      std::make_shared<Shape>(target_shape), std::make_shared<Stride>(target_stride),
      input->dtype()->data_type(), device, storage_offset);

  CHECK_OR_RETURN(JUST(input->has_eager_blob_object()));
  // new output tensor
  const auto& blob_object = JUST(input->eager_blob_object());
  bool requires_grad = (autograd::GradMode::is_enabled() && input->requires_grad());
  auto tensor_impl = std::make_shared<EagerLocalTensorImpl>(
      tensor_meta, JUST(input->tensor_storage()), requires_grad,
      /*is_leaf=*/!requires_grad);
  JUST(tensor_impl->InitEagerBlobObject(JUST(blob_object->compute_local_dep_object())));

  auto view_tensor = std::make_shared<LocalTensor>(tensor_impl);

  const std::shared_ptr<vm::EagerBlobObject>& view_eager_blob_object =
      JUST(view_tensor->eager_blob_object());
  view_eager_blob_object->set_storage_offset(JUST(view_tensor->storage_offset()));
  return std::static_pointer_cast<Tensor>(view_tensor);
}

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& target_shape) {
  Stride target_stride(target_shape);
  return Reshape(input, target_shape, target_stride);
}

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                      const Stride& target_stride) {
  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, target_shape, target_stride, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    Shape input_shape(input->shape()->dim_vec());
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
      in_grads->resize(1);
      JUST(oneflow::VectorAt(*in_grads, 0)) =
          JUST(functional::Reshape(JUST(oneflow::VectorAt(out_grads, 0)), input_shape));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return false; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::reshape_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> Slice(const std::shared_ptr<Tensor>& input, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& steps) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = starts.size();

  CHECK_OR_RETURN(ndim == shape->NumAxes())
      << Error::RuntimeError() << "view::Slice(): starts size is expected " << shape->NumAxes()
      << ", but got " << ndim;

  CHECK_OR_RETURN(ends.size() == ndim && steps.size() == ndim)
      << Error::RuntimeError() << "view::Slice(): " << (ends.size() != ndim ? "ends" : "steps")
      << " size is not equal to start.";

  DimVector target_dims(ndim);
  Stride target_strides(ndim);
  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  for (int i = 0; i < ndim; ++i) {
    int64_t step = std::min(steps[i], shape->At(i));
    CHECK_OR_RETURN(step >= 0) << Error::RuntimeError() << "Step must be greater than zero.";
    int64_t start = std::min(starts[i], shape->At(i));
    int64_t end = std::min(ends[i], shape->At(i));
    if (start < 0) { start += shape->At(i); }
    if (start < 0) start = 0;
    if (end < 0) { end += shape->At(i); }
    if (end < start) end = start;
    int64_t length = start == end ? 0 : (end - start + step - 1) / step;
    target_dims[i] = length;
    target_strides[i] = step * strides->at(i);
    storage_offset += start * strides->at(i);
  }

  auto output = JUST(BasicView(input, Shape(target_dims), target_strides, storage_offset));
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    const Shape in_shape = *input->shape();
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
      in_grads->resize(1);
      (*in_grads)[0] = JUST(functional::SliceGrad(out_grads[0], in_shape, starts, ends, steps));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::slice_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> Unsqueeze(const std::shared_ptr<Tensor>& input, const int32_t& expand_dim) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const auto& ndim = shape->NumAxes();

  DimVector target_dim_vec(ndim + 1);
  Stride target_stride_vec(ndim + 1);

  {
    int cnt = 0;
    for (int i = 0; i < ndim; i++) {
      if (i == expand_dim) { cnt++; }
      target_dim_vec[cnt] = shape->At(i);
      target_stride_vec[cnt] = strides->at(i);
      cnt++;
    }
    target_dim_vec[expand_dim] = 1;
    target_stride_vec[expand_dim] = expand_dim < ndim ? strides->at(expand_dim) : 1;
  }

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, Shape(target_dim_vec), target_stride_vec, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
      in_grads->resize(1);
      JUST(oneflow::VectorAt(*in_grads, 0)) =
          JUST(functional::Reshape(JUST(oneflow::VectorAt(out_grads, 0)), *shape));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return false; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::unsqueeze_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> Squeeze(const std::shared_ptr<Tensor>& input,
                      const std::vector<int32_t>& squeeze_dims) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();

  const int target_ndim = ndim - squeeze_dims.size();
  DimVector target_dim_vec(target_ndim);
  Stride target_stride_vec(target_ndim);

  {
    int cnt = 0;
    for (int i = 0; i < ndim; i++) {
      if (find(squeeze_dims.begin(), squeeze_dims.end(), i) == squeeze_dims.end()) {
        target_dim_vec[cnt] = shape->At(i);
        target_stride_vec[cnt] = strides->at(i);
        cnt++;
      }
    }
  }

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, Shape(target_dim_vec), target_stride_vec, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
      in_grads->resize(1);
      JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::Reshape(
          JUST(oneflow::VectorAt(out_grads, 0)), Shape(input->shape()->dim_vec())));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::squeeze_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> Expand(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& in_shape,
                     const std::vector<int32_t>& expand_shape) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = in_shape.size();

  const int64_t target_ndim = expand_shape.size();
  DimVector target_dim_vec(target_ndim);
  Stride target_stride_vec(target_ndim);

  for (int i = 0; i < target_ndim; i++) {
    if (i < ndim) {
      if (expand_shape[target_ndim - 1 - i] == -1) {
        target_dim_vec[target_ndim - 1 - i] = in_shape[ndim - 1 - i];
        target_stride_vec[target_ndim - 1 - i] = strides->at(ndim - 1 - i);
      } else if (in_shape[ndim - 1 - i]
                 == 1) {  // TODO (bowen): what if dim is 1, should stride be set to 0?
        target_dim_vec[target_ndim - 1 - i] = expand_shape[target_ndim - 1 - i];
        target_stride_vec[target_ndim - 1 - i] = 0;
      } else {
        if (expand_shape[target_ndim - 1 - i] != in_shape[ndim - 1 - i]) {
          return Error::RuntimeError()
                 << "The expanded size of the tensor (" << expand_shape[target_ndim - 1 - i] << ")"
                 << "must match the existing size (" << in_shape[ndim - 1 - i]
                 << ") at non-singleton dimension " << ndim - i << ".  Target sizes: "
                 << ".  Tensor sizes: " << shape->ToString();
        }
        target_dim_vec[target_ndim - 1 - i] = in_shape[ndim - 1 - i];
        target_stride_vec[target_ndim - 1 - i] = strides->at(ndim - 1 - i);
      }
    } else {
      if (expand_shape[target_ndim - 1 - i] == -1) {
        return Error::RuntimeError() << "The expanded size of the tensor (-1) "
                                     << "isn't allowed in a leading, non-existing dimension 0";
      }
      target_dim_vec[target_ndim - 1 - i] = expand_shape[target_ndim - 1 - i];
      target_stride_vec[target_ndim - 1 - i] = 0;
    }
  }

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, Shape(target_dim_vec), target_stride_vec, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      (*in_grads)[0] = JUST(functional::ExpandGrad(out_grads[0], in_shape, expand_shape));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::expand_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> Narrow(const std::shared_ptr<Tensor>& input, const int64_t& dim, const int64_t& start,
                     const int64_t& length) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), shape->dim_vec().cbegin(), shape->dim_vec().cbegin() + dim);
  dim_vec.insert(dim_vec.end(), length);
  dim_vec.insert(dim_vec.end(), shape->dim_vec().cbegin() + dim + 1, shape->dim_vec().end());

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  Shape target_shape(dim_vec);

  Stride stride(ndim);
  for (int i = 0; i < ndim; ++i) {
    stride[i] = strides->at(i);
    if (dim == i) { storage_offset += start * strides->at(i); }
  }

  auto output = JUST(BasicView(input, target_shape, stride, storage_offset));
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      auto like = JUST(functional::Empty(Shape(input->shape()->dim_vec()), input->dtype(),
                                         JUST(input->device()), /*pin_memory=*/false));
      in_grads->resize(1);
      (*in_grads)[0] = JUST(functional::NarrowGrad(out_grads[0], like, dim, start, length));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::narrow_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> AsStrided(const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& size,
                        const std::vector<int32_t>& stride_vec, const int32_t& storage_offset) {
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), size.begin(), size.end());
  Shape target_shape(dim_vec);
  Stride stride(stride_vec.begin(), stride_vec.end());
  auto output = JUST(view::BasicView(input, target_shape, stride, storage_offset));
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      auto like = JUST(functional::Empty(Shape(input->shape()->dim_vec()), input->dtype(),
                                         JUST(input->device()), /*pin_memory=*/false));
      in_grads->resize(1);
      (*in_grads)[0] =
          JUST(functional::AsStridedGrad(out_grads[0], like, size, stride_vec, storage_offset));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::as_strided_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> Transpose(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& permute) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());

  CHECK_EQ_OR_RETURN(permute.size(), ndim)
      << "permute size should be equal to input tensor's ndim, but got " << permute.size();
  auto positive_perm = permute;
  for (auto i = 0; i < positive_perm.size(); i++) { JUST(maybe_wrap_dim(positive_perm[i], ndim)); }

  DimVector target_dims(ndim);
  Stride stride(ndim);
  for (int i = 0; i < ndim; ++i) {
    target_dims[i] = shape->At(permute[i]);
    stride[i] = strides->at(permute[i]);
  }

  auto output = JUST(BasicView(input, Shape(target_dims), stride, storage_offset));
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      std::vector<int32_t> grad_perm;
      grad_perm.resize(ndim);
      for (int i = 0; i < ndim; ++i) { grad_perm[permute[i]] = i; }
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      (*in_grads)[0] = JUST(functional::Transpose(out_grads[0], grad_perm));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::transpose_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<Tensor> UnfoldTensor(const std::shared_ptr<Tensor>& input, const int32_t& dimension,
                           const int32_t& size, const int32_t& step) {
  const auto& shape = input->shape();
  const auto& stride = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());

  CHECK_GE_OR_RETURN(dimension, 0) << "attibute dimension should be >= 0, but got " << dimension;
  CHECK_LE_OR_RETURN(dimension, ndim)
      << "attibute dimension should be <= input tensor's ndim, but got " << dimension;

  const int32_t max_size = ndim == 0 ? 1 : shape->At(dimension);
  CHECK_GT_OR_RETURN(size, 0) << "attibute size should be > 0, but got " << size;
  CHECK_LE_OR_RETURN(size, max_size)
      << "attibute size should be <= max_size(" << max_size << ") but got " << size;
  CHECK_GT_OR_RETURN(step, 0) << "attibute step should be > 0, but got " << size;

  DimVector out_shape(ndim + 1);
  Stride out_stride(ndim + 1);
  out_shape[ndim] = size;
  out_stride[ndim] = ndim == 0 ? 1 : stride->at(dimension);
  for (int64_t d = 0; d < ndim; ++d) {
    const int64_t in_size_at_d = shape->At(d);
    if (d == dimension) {
      out_shape.at(d) = (in_size_at_d - size) / step + 1;
      out_stride.at(d) = step * stride->at(d);
    } else {
      out_shape.at(d) = in_size_at_d;
      out_stride.at(d) = stride->at(d);
    }
  }
  auto output = JUST(BasicView(input, Shape(out_shape), out_stride, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      (*in_grads)[0] =
          JUST(functional::UnfoldTensorGrad(out_grads[0], input, dimension, size, step));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::unfold_tensor_backward", backward_fn,
                                                 {input}, &outputs));
  }

  return output;
}

Maybe<Tensor> Diagonal(const std::shared_ptr<Tensor>& input, const int32_t offset,
                       const int32_t dim1, const int32_t dim2) {
  const auto& shape = input->shape();
  const auto& stride = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());

  // infer output storage_offset
  int64_t diag_size = 0;
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(shape->At(dim1), shape->At(dim2) - offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(shape->At(dim1) + offset, shape->At(dim2)), 0);
  }
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * stride->at(dim2);
  } else {
    storage_offset -= offset * stride->at(dim1);
  }

  CHECK_GE_OR_RETURN(ndim, 2) << "input tensor's ndim should be >= 2, but got " << ndim;
  // infer output shape and stride
  DimVector out_shape(shape->dim_vec());
  Stride out_stride(*stride);
  out_shape.erase(out_shape.begin() + std::max(dim1, dim2));
  out_stride.erase(out_stride.begin() + std::max(dim1, dim2));
  out_shape.erase(out_shape.begin() + std::min(dim1, dim2));
  out_stride.erase(out_stride.begin() + std::min(dim1, dim2));
  out_shape.emplace_back(diag_size);
  out_stride.emplace_back(stride->at(dim1) + stride->at(dim2));

  // generate view tensor
  auto output = JUST(BasicView(input, Shape(out_shape), out_stride, storage_offset));
  // autograd
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    std::vector<int32_t> input_index{dim1, dim2};
    for (int32_t i = 0; i < ndim; i++) {
      if (i != dim1 && i != dim2) { input_index.push_back(i); }
    }

    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      std::shared_ptr<one::Tensor> d_x = JUST(functional::Transpose(input, input_index));
      (*in_grads)[0] = JUST(functional::DiagonalGrad(out_grads[0], d_x, offset));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::diagonal_backward", backward_fn, {input},
                                                 &outputs));
  }

  return output;
}

}  // namespace view
}  // namespace one
}  // namespace oneflow
