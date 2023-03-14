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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/common/wrap_dim_utils.h"
#include "oneflow/core/functional/functional_api.yaml.h"

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

static bool IsOverlappingMemorys(const std::vector<int64_t>& sizes,
                                 const std::vector<int64_t>& strides) {
  // reference: torch/csrc/autograd/FunctionsManual.cpp _maybe_overlapping_memory()
  if (sizes.size() > 0) {
    std::vector<std::size_t> argsort(sizes.size());
    std::iota(argsort.begin(), argsort.end(), 0);
    std::sort(argsort.begin(), argsort.end(),
              [&](std::size_t i, std::size_t j) { return strides[i] < strides[j]; });
    int64_t max_index_in_slice = 0;
    for (auto i : argsort) {
      auto stride_ = strides[i];
      if (stride_ <= max_index_in_slice) { return true; }
      max_index_in_slice += stride_ * (sizes[i] - 1);
    }
  }
  return false;
}

static int64_t MinStorageSize(const std::vector<int64_t>& sizes,
                              const std::vector<int64_t>& strides, int64_t storage_offset) {
  int64_t storage_size = storage_offset + 1;
  int64_t ndim = sizes.size();
  for (size_t i = 0; i < ndim; i++) {
    auto size_i = sizes[i];
    if (size_i == 0) { return storage_offset; }
    storage_size += (size_i - 1) * strides[i];
  }
  return storage_size;
}

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const int64_t storage_offset) {
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
                        const Stride& target_stride, const int64_t storage_offset) {
  auto device = JUST(input->device());
  auto tensor_meta =
      SymbolOf(LocalTensorMeta(target_shape, target_stride, input->dtype()->data_type(), device));

  CHECK_OR_RETURN(JUST(input->has_eager_blob_object()));
  // new output tensor
  const auto& blob_object = JUST(input->eager_blob_object());
  bool requires_grad = (autograd::GradMode::is_enabled() && input->requires_grad());
  auto tensor_impl = std::make_shared<EagerLocalTensorImpl>(JUST(input->tensor_storage()),
                                                            storage_offset, requires_grad,
                                                            /*is_leaf=*/!requires_grad);
  JUST(
      tensor_impl->InitEagerBlobObject(tensor_meta, JUST(blob_object->compute_local_dep_object())));

  auto view_tensor = std::make_shared<LocalTensor>(tensor_impl);

  const std::shared_ptr<vm::EagerBlobObject>& view_eager_blob_object =
      JUST(view_tensor->eager_blob_object());
  view_eager_blob_object->set_storage_offset(JUST(view_tensor->storage_offset()));
  return std::static_pointer_cast<Tensor>(view_tensor);
}

Maybe<void> InplaceView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const Stride& target_stride, const int64_t storage_offset) {
  Symbol<LocalTensorMeta> new_tensor_meta = SymbolOf(LocalTensorMeta(
      target_shape, target_stride, input->dtype()->data_type(), JUST(input->device())));

  bool requires_grad = (autograd::GradMode::is_enabled() && input->requires_grad());
  std::shared_ptr<EagerLocalTensorImpl> new_tensor_impl = std::make_shared<EagerLocalTensorImpl>(
      JUST(input->tensor_storage()), storage_offset, /*requires_grad=*/requires_grad,
      /*is_leaf=*/!requires_grad);
  JUST(new_tensor_impl->InitEagerBlobObject(
      new_tensor_meta, JUST(JUST(input->eager_blob_object())->compute_local_dep_object())));
  JUST(JUST(input->AsLocalTensor())->set_impl(new_tensor_impl));
  return Maybe<void>::Ok();
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

Maybe<Tensor> Unsqueeze(const std::shared_ptr<Tensor>& input, const int32_t expand_dim) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const auto& ndim = shape->NumAxes();

  DimVector target_dim_vec(ndim + 1);
  Stride target_stride_vec(ndim + 1);

  {
    int cnt = 0;
    for (int i = 0; i < ndim; i++) {
      if (i == expand_dim) { cnt++; }
      target_dim_vec[cnt] = shape->at(i);
      target_stride_vec[cnt] = strides->at(i);
      cnt++;
    }
    target_dim_vec[expand_dim] = 1;
    target_stride_vec[expand_dim] =
        expand_dim < ndim ? strides->at(expand_dim) * target_dim_vec.at(expand_dim + 1) : 1;
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

Maybe<void> InplaceUnsqueeze(const std::shared_ptr<Tensor>& input, const int32_t expand_dim) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const auto& ndim = shape->NumAxes();

  DimVector target_dim_vec(ndim + 1);
  Stride target_stride_vec(ndim + 1);

  {
    int cnt = 0;
    for (int i = 0; i < ndim; i++) {
      if (i == expand_dim) { cnt++; }
      target_dim_vec[cnt] = shape->at(i);
      target_stride_vec[cnt] = strides->at(i);
      cnt++;
    }
    target_dim_vec[expand_dim] = 1;
    target_stride_vec[expand_dim] =
        expand_dim < ndim ? strides->at(expand_dim) * target_dim_vec.at(expand_dim + 1) : 1;
  }

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  JUST(view::InplaceView(input, Shape(target_dim_vec), target_stride_vec, storage_offset));

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
    TensorTuple outputs{input};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::inplace_unsqueeze_backward", backward_fn,
                                                 {input}, &outputs));
  }
  return Maybe<void>::Ok();
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

Maybe<void> InplaceSqueeze(const std::shared_ptr<Tensor>& input,
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
  JUST(view::InplaceView(input, Shape(target_dim_vec), target_stride_vec, storage_offset));

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
    TensorTuple outputs{input};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::inplace_squeeze_backward", backward_fn,
                                                 {input}, &outputs));
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> Expand(const std::shared_ptr<Tensor>& input, const Shape& expand_shape) {
  const Shape& input_shape = *input->shape();
  const Stride& input_stride = *JUST(input->stride());
  size_t lpad = expand_shape.size() - input_shape.size();
  CHECK_GE_OR_RETURN(lpad, 0);  // NOLINT(maybe-need-error-msg)

  Stride expand_stride(expand_shape.size(), 0);
  std::vector<int32_t> reduce_dims;
  reduce_dims.reserve(expand_shape.size());

  for (int i = expand_shape.size() - 1; i >= 0; --i) {
    int64_t dim = i < lpad ? 1 : input_shape[i - lpad];
    if (dim == expand_shape[i]) {
      if (i >= lpad) {
        expand_stride[i] = input_stride[i - lpad];
      } else if (i < expand_shape.size() - 1) {
        expand_stride[i] = expand_stride[i + 1] * expand_shape[i + 1];
      }
    } else {
      CHECK_EQ_OR_RETURN(dim, 1);  // NOLINT(maybe-need-error-msg)
      reduce_dims.push_back(i);
    }
  }

  if (input_shape.size() == 0) {
    // handle scalar expand backward reduce dims
    reduce_dims.clear();
    for (int32_t axis = 0; axis < expand_shape.size(); ++axis) { reduce_dims.push_back(axis); }
  }

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, expand_shape, expand_stride, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      in_grads->at(0) = out_grads[0];
      bool keep_dims = (input_shape.size() > 0);
      if (reduce_dims.size() > 0) {
        in_grads->at(0) = JUST(functional::ReduceSum(in_grads->at(0), reduce_dims, keep_dims));
      }
      if (lpad > 0 && keep_dims) {
        in_grads->at(0) = JUST(functional::Flatten(in_grads->at(0), 0, lpad));
      }
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::expand_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<void> InplaceExpand(const std::shared_ptr<Tensor>& input, const Shape& expand_shape) {
  const Shape& input_shape = *input->shape();
  const Stride& input_stride = *JUST(input->stride());
  size_t lpad = expand_shape.size() - input_shape.size();
  CHECK_GE_OR_RETURN(lpad, 0);  // NOLINT(maybe-need-error-msg)

  Stride expand_stride(expand_shape.size(), 0);
  std::vector<int32_t> reduce_dims;
  reduce_dims.reserve(expand_shape.size());

  for (int i = expand_shape.size() - 1; i >= 0; --i) {
    int64_t dim = i < lpad ? 1 : input_shape[i - lpad];
    if (dim == expand_shape[i]) {
      if (i >= lpad) {
        expand_stride[i] = input_stride[i - lpad];
      } else if (i < expand_shape.size() - 1) {
        expand_stride[i] = expand_stride[i + 1] * expand_shape[i + 1];
      }
    } else {
      CHECK_EQ_OR_RETURN(dim, 1);  // NOLINT(maybe-need-error-msg)
      reduce_dims.push_back(i);
    }
  }

  if (input_shape.size() == 0) {
    // handle scalar expand backward reduce dims
    reduce_dims.clear();
    for (int32_t axis = 0; axis < expand_shape.size(); ++axis) { reduce_dims.push_back(axis); }
  }

  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());
  JUST(view::InplaceView(input, expand_shape, expand_stride, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      in_grads->at(0) = out_grads[0];
      bool keep_dims = (input_shape.size() > 0);
      if (reduce_dims.size() > 0) {
        in_grads->at(0) = JUST(functional::ReduceSum(in_grads->at(0), reduce_dims, keep_dims));
      }
      if (lpad > 0 && keep_dims) {
        in_grads->at(0) = JUST(functional::Flatten(in_grads->at(0), 0, lpad));
      }
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{input};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::expand_backward", backward_fn, {input},
                                                 &outputs));
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> Narrow(const std::shared_ptr<Tensor>& input, const int64_t dim, const int64_t start,
                     const int64_t length) {
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
      auto like =
          JUST(functional::Empty(Shape(input->shape()->dim_vec()), input->dtype(),
                                 JUST(input->device()), /*requires_grad=*/input->requires_grad(),
                                 /*pin_memory=*/false));
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

Maybe<Tensor> AsStridedGrad(const std::shared_ptr<one::Tensor>& dy,
                            const std::shared_ptr<one::Tensor>& input,
                            const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                            const int64_t storage_offset) {
  CHECK_OR_RETURN(input->is_local()) << "input must be local tensor.";
  // reference: torch/csrc/autograd/FunctionsManual.cpp
  const size_t odim = dy->ndim();
  std::vector<int64_t> out_sizes_, out_strides_;
  out_sizes_.reserve(odim);
  out_strides_.reserve(odim);
  auto grad = dy;
  for (int64_t i = odim - 1; i >= 0; i--) {
    auto size_i = sizes[i];
    auto stride_i = strides[i];
    if (size_i == 0) {
      return functional::Constant(*dy->shape(), 0, grad->dtype(), JUST(grad->device()));
    } else if (size_i == 1) {
      grad = JUST(functional::Squeeze(grad, std::vector<int32_t>{int(i)}));
    } else if (stride_i == 0) {
      grad = JUST(functional::ReduceSum(grad, std::vector<int32_t>{int(i)}, false));
    } else {
      out_sizes_.insert(out_sizes_.begin(), size_i);
      out_strides_.insert(out_strides_.begin(), stride_i);
    }
  }

  // Step (2)~(4) for the algorithm in NOTE [ Detecting Memory Overlap Within A
  // Strided Tensor ]
  //              on output geometry
  const bool out_maybe_overlap = IsOverlappingMemorys(out_sizes_, out_strides_);

  // For input geometry,
  //   check for size 0 dimensions,
  //   skip size 1 dimensions,
  // Step (0)~(1) for the algorithm in NOTE [ Detecting Memory Overlap Within A
  // Strided Tensor ]
  //              on input geometry
  auto idim = input->ndim();
  std::vector<int64_t> inp_sizes(input->shape()->begin(), input->shape()->end());
  std::vector<int64_t> inp_strides(JUST(input->stride())->begin(), JUST(input->stride())->end());
  std::vector<int64_t> inp_sizes_, inp_strides_;
  inp_sizes_.reserve(idim);
  inp_strides_.reserve(idim);
  for (int64_t i = idim - 1; i >= 0; i--) {
    auto size_i = inp_sizes[i];
    auto stride_i = inp_strides[i];
    if (size_i == 0) {
      return functional::Constant(*input->shape(), 0, grad->dtype(), JUST(grad->device()));
    } else if (size_i != 1) {
      inp_sizes_.insert(inp_sizes_.begin(), size_i);
      inp_strides_.insert(inp_strides_.begin(), stride_i);
    }
  }
  // Step (1)~(4) for the algorithm in NOTE [ Detecting Memory Overlap Within A
  // Strided Tensor ]
  //              on input geometry
  const bool inp_maybe_overlap = IsOverlappingMemorys(inp_sizes_, inp_strides_);

  // Rest of this function implements
  // Step (1)~(4) for the algorithm in NOTE [ as_strided Backward and
  // layout-aware/agnostic autograd ]
  // TODO: Raise if not all output values are visible in input geometry.
  //       Technically speaking, if you treat those values as constants, not
  //       raising is fine, and mathematically correct. However, these values
  //       really are contained in some base tensor, and by treating them as
  //       constants we are ignoring this tight dependency. Therefore, it is
  //       more sensible to raise here.

  // Step (1): create underlying tensor as "storage"
  auto input_storage_offset = JUST(input->storage_offset());
  auto shared_offset = std::min(input_storage_offset, storage_offset);
  auto inp_effective_offset = input_storage_offset - shared_offset;
  auto out_effective_offset = storage_offset - shared_offset;
  auto base_size = std::max(MinStorageSize(inp_sizes_, inp_strides_, inp_effective_offset),
                            MinStorageSize(out_sizes_, out_strides_, out_effective_offset));
  auto storage =
      JUST(functional::Constant(Shape({base_size}), 0, grad->dtype(), JUST(grad->device())));

  std::shared_ptr<Tensor> flatten_full_indices;
  if (inp_maybe_overlap || out_maybe_overlap) {
    flatten_full_indices = JUST(functional::Arange(Scalar(0), Scalar(base_size), Scalar(1),
                                                   DType::Int64(), JUST(grad->device())));
  }

  // Step (2): use output geometry to scatter gradients into storage
  if (out_maybe_overlap) {
    auto out_indices = JUST(functional::AsStrided(flatten_full_indices, out_sizes_, out_strides_,
                                                  out_effective_offset));
    storage = JUST(functional::IndexAddInplace(
        storage, 0,
        JUST(functional::Reshape(out_indices, Shape({out_indices->shape()->elem_cnt()}))),
        JUST(functional::Reshape(grad, Shape({grad->shape()->elem_cnt()}))), Scalar(1.0)));
  } else {
    // assume that new tensors have 0 storage offset
    // torch impl: storage.as_strided(out_sizes_, out_strides_, out_effective_offset)
    //     .copy_(grad);
    // TODO(wangyinggang): use functional::copy_ replace this TensorSetItem
    storage = JUST(functional::AsStrided(storage, out_sizes_, out_strides_, out_effective_offset));
    functional::TensorIndex ellipsis_index;
    ellipsis_index.emplace_back(functional::detail::EllipsisIndex());
    JUST(functional::TensorSetItem(storage, ellipsis_index, grad));
  }

  // Step (3): if input tensor has overlapping memory, divide scattered gradient
  //           at storage[i] by the number of times i shows up in input geometry
  if (inp_maybe_overlap) {
    auto count =
        JUST(functional::Constant(*storage->shape(), 0, storage->dtype(), JUST(storage->device())));
    flatten_full_indices = JUST(functional::AsStrided(flatten_full_indices, inp_sizes_,
                                                      inp_strides_, inp_effective_offset));
    auto inp_indices = JUST(functional::Reshape(
        flatten_full_indices, Shape({flatten_full_indices->shape()->elem_cnt()})));

    auto ones = JUST(functional::Constant(Shape({1}), 0, grad->dtype(), JUST(grad->device())));
    count = JUST(functional::IndexAddInplace(count, 0, inp_indices, ones, Scalar(1.0)));
    count = JUST(functional::Expand(count, *inp_indices->shape()));
    storage = JUST(functional::Div(storage, count));  // this will give nan outside visible range
  }

  // Step (4): return as_strided view of the storage tensor with input geometry
  return functional::AsStrided(storage, inp_sizes, inp_strides, inp_effective_offset);
}

Maybe<Tensor> AsStrided(const std::shared_ptr<one::Tensor>& input,
                        const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                        const int64_t storage_offset) {
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), sizes.begin(), sizes.end());
  Shape target_shape(dim_vec);
  Stride stride(strides.begin(), strides.end());
  auto output = JUST(view::BasicView(input, target_shape, stride, storage_offset));
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      (*in_grads)[0] = JUST(AsStridedGrad(out_grads[0], input, sizes, strides, storage_offset));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::as_strided_backward", backward_fn, {input},
                                                 &outputs));
  }
  return output;
}

Maybe<void> InplaceAsStrided(const std::shared_ptr<one::Tensor>& input,
                             const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                             const int64_t storage_offset) {
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), sizes.begin(), sizes.end());
  Shape target_shape(dim_vec);
  Stride stride(strides.begin(), strides.end());
  JUST(view::InplaceView(input, target_shape, stride, storage_offset));
  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      CHECK_EQ_OR_RETURN(out_grads.size(), 1)
          << "out grad size should be 1, but got " << out_grads.size();
      in_grads->resize(1);
      (*in_grads)[0] = JUST(AsStridedGrad(out_grads[0], input, sizes, strides, storage_offset));
      return Maybe<void>::Ok();
    };
    backward_fn->status = []() { return true; };
    TensorTuple outputs{input};
    JUST(GetThreadLocalAutogradEngine()->AddNode("view::inplace_as_strided_backward", backward_fn,
                                                 {input}, &outputs));
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> Transpose(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& permute) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsLocalTensor())->storage_offset());

  CHECK_EQ_OR_RETURN(permute.size(), ndim)
      << "permute size should be equal to input tensor's ndim, but got " << permute.size();
  auto positive_perm = permute;
  for (auto i = 0; i < positive_perm.size(); i++) {
    positive_perm[i] = JUST(maybe_wrap_dim(positive_perm[i], ndim));
  }

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

Maybe<Tensor> UnfoldTensor(const std::shared_ptr<Tensor>& input, const int32_t dimension,
                           const int32_t size, const int32_t step) {
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

Maybe<void> Touch(std::shared_ptr<Tensor> input, Symbol<Stream> stream) {
  auto eager_blob_objects = std::make_shared<vm::EagerBlobObjectList>();
  if (input->is_global()) { input = JUST(input->cur_rank_phy_tensor()); }
  if (input) { eager_blob_objects->push_back(JUST(input->eager_blob_object())); }
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->TouchTensors(eager_blob_objects, stream);
  }));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
