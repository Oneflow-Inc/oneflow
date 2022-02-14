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
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace oneflow {
namespace one {
namespace view {

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
  auto tensor_meta = std::make_shared<MirroredTensorMeta>(
      std::make_shared<Shape>(target_shape), input->dtype()->data_type(), device,
      std::make_shared<Stride>(target_stride), storage_offset);

  CHECK_OR_RETURN(JUST(input->has_eager_blob_object()));
  // new output tensor
  const auto& blob_object = JUST(input->eager_blob_object());
  auto tensor_impl = std::make_shared<EagerMirroredTensorImpl>(
      tensor_meta, JUST(input->tensor_storage()), input->requires_grad(),
      /*is_leaf=*/!input->requires_grad());
  JUST(tensor_impl->InitEagerBlobObject(JUST(blob_object->compute_local_dep_object())));
  std::shared_ptr<Tensor> output(new MirroredTensor(tensor_impl));
  // run tensor view instruction
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->TensorView(JUST(input->AsMirroredTensor()), JUST(output->AsMirroredTensor()));
  }));
  return output;
}

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& target_shape) {
  if (!(input->is_eager() && input->is_local())) {
    return Error::RuntimeError() << "view::Reshape(): input should be eager local tensor, but got "
                                 << (input->is_lazy() ? "lazy" : "consistent");
  }

  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
  std::shared_ptr<Tensor> output = JUST(BasicView(input, target_shape, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    Shape input_shape(input->shape()->dim_vec());
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) = JUST(functional::Reshape(out_grads.at(0), input_shape));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::reshape_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> Squeeze(const std::shared_ptr<Tensor>& input,
                      const std::vector<int32_t>& squeeze_dims) {
  if (!(input->is_eager() && input->is_local())) {
    return Error::RuntimeError() << "view::Squeeze(): input should be eager local tensor, but got "
                                 << (input->is_lazy() ? "lazy" : "consistent");
  }

  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();

  const int target_ndim = ndim - squeeze_dims.size();
  DimVector target_dim_vec(target_ndim);
  StrideVector target_stride_vec(target_ndim);

  int cnt = 0;
  for (int i = 0; i < ndim; i++) {
    if (find(squeeze_dims.begin(), squeeze_dims.end(), i) == squeeze_dims.end()) {
      target_dim_vec[cnt] = shape->At(i);
      target_stride_vec[cnt] = strides->At(i);
      cnt++;
    }
  }

  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, Shape(target_dim_vec), Stride(target_stride_vec), storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) = JUST(functional::ReshapeLike(out_grads.at(0), input));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::squeeze_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> ExpandDims(const std::shared_ptr<Tensor>& input, const int32_t& expand_dim) {
  if (!(input->is_eager() && input->is_local())) {
    return Error::RuntimeError()
           << "view::ExpandDims(): input should be eager local tensor, but got "
           << (input->is_lazy() ? "lazy" : "consistent");
  }

  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const auto& ndim = shape->NumAxes();

  DimVector target_dim_vec(ndim + 1);
  StrideVector target_stride_vec(ndim + 1);

  int cnt = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == expand_dim) { cnt++; }
    target_dim_vec[cnt] = shape->At(i);
    target_stride_vec[cnt] = strides->At(i);
    cnt++;
  }
  target_dim_vec[expand_dim] = 1;
  target_stride_vec[expand_dim] = strides->At(expand_dim);

  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, Shape(target_dim_vec), Stride(target_stride_vec), storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) = JUST(functional::Reshape(out_grads.at(0), *shape));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::expanddims_backward",
                                                            backward_fn, {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> Expand(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& in_shape,
                     const std::vector<int32_t>& expand_shape) {
  if (!(input->is_eager() && input->is_local())) {
    return Error::RuntimeError() << "view::Expand(): input should be eager local tensor, but got "
                                 << (input->is_lazy() ? "lazy" : "consistent");
  }

  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = in_shape.size();

  const int64_t target_ndim = expand_shape.size();
  DimVector target_dim_vec(target_ndim);
  StrideVector target_stride_vec(target_ndim);

  for (int i = 0; i < target_ndim; i++) {
    if (i < ndim) {
      if (expand_shape[target_ndim - 1 - i] == -1) {
        target_dim_vec[target_ndim - 1 - i] = in_shape[ndim - 1 - i];
        target_stride_vec[target_ndim - 1 - i] = strides->At(ndim - 1 - i);
      } else if (in_shape[ndim - 1 - i]
                 == 1) {  // TODO (bowen): what if dim is 1, should stride be set to 0?
        target_dim_vec[target_ndim - 1 - i] = expand_shape[target_ndim - 1 - i];
        target_stride_vec[target_ndim - 1 - i] = 0;
      } else {
        if (expand_shape[target_ndim - 1 - i] != in_shape[ndim - 1 - i]) {
          return Error::RuntimeError()
                 << "view::Expand(): The expanded size of the tensor ("
                 << expand_shape[target_ndim - 1 - i] << ")"
                 << "must match the existing size (" << in_shape[ndim - 1 - i]
                 << ") at non-singleton dimension " << ndim - i << ".  Target sizes: "
                 << ".  Tensor sizes: " << shape->ToString();
        }
        target_dim_vec[target_ndim - 1 - i] = in_shape[ndim - 1 - i];
        target_stride_vec[target_ndim - 1 - i] = strides->At(ndim - 1 - i);
      }
    } else {
      if (expand_shape[target_ndim - 1 - i] == -1) {
        return Error::RuntimeError() << "view::Expand(): The expanded size of the tensor (-1) "
                                     << "isn't allowed in a leading, non-existing dimension 0";
      }
      target_dim_vec[target_ndim - 1 - i] = expand_shape[target_ndim - 1 - i];
      target_stride_vec[target_ndim - 1 - i] = 0;
    }
  }

  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, Shape(target_dim_vec), Stride(target_stride_vec), storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) =
                  JUST(functional::ExpandGrad(out_grads.at(0), in_shape, expand_shape));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::expand_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> Slice(const std::shared_ptr<Tensor>& input, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& steps) {
  
  CHECK_OR_RETURN(input->is_eager() && input->is_local())
      << Error::RuntimeError() << "view::Slice(): input should be eager local tensor, but is "
      << (input->is_lazy() ? "lazy" : "consistent");
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
  StrideVector target_strides(ndim);
  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
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
    target_strides[i] = step * strides->At(i);
    storage_offset += start * strides->At(i);
  }

  auto output = JUST(BasicView(input, Shape(target_dims), Stride(target_strides), storage_offset));
  if (input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              (*in_grads)[0] = JUST(functional::SliceGrad(JUST(VectorAt(out_grads, 0)),
                                                          Shape(input->shape()->dim_vec()), starts,
                                                          ends, steps));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::slice_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> Narrow(const std::shared_ptr<Tensor>& input, const int64_t& dim, const int64_t& start,
                     const int64_t& length) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  CHECK_GT_OR_RETURN(ndim, 0);
  CHECK_GE_OR_RETURN(dim, 0);
  CHECK_GE_OR_RETURN(start, 0);
  CHECK_GE_OR_RETURN(length, 0);
  CHECK_GE_OR_RETURN(shape->At(dim), start + length);

  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), shape->dim_vec().cbegin(), shape->dim_vec().cbegin() + dim);
  dim_vec.insert(dim_vec.end(), length);
  dim_vec.insert(dim_vec.end(), shape->dim_vec().cbegin() + dim + 1, shape->dim_vec().end());

  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
  Shape target_shape(dim_vec);

  StrideVector stride_vec(ndim);
  for (int i = 0; i < ndim; ++i) {
    stride_vec[i] = strides->At(i);
    if (dim == i) { storage_offset += start * strides->At(i); }
  }

  auto output = JUST(BasicView(input, target_shape, Stride(stride_vec), storage_offset));
  if (input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              auto like = JUST(functional::Empty(Shape(input->shape()->dim_vec()), input->dtype(),
                                                 JUST(input->device())));
              in_grads->resize(1);
              in_grads->at(0) =
                  JUST(functional::NarrowGrad(out_grads.at(0), like, dim, start, length));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::narrow_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

void CheckIsPerm(const std::vector<int32_t>& perm) {
  std::vector<bool> is_used(perm.size(), false);
  FOR_RANGE(size_t, i, 0, perm.size()) {
    CHECK_GE(perm[i], 0);
    CHECK_LE(perm[i], perm.size());
    CHECK_EQ(is_used[perm[i]], false);
    is_used[perm[i]] = true;
  }
}

Maybe<Tensor> Transpose(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& permute) {
  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());

  CHECK_EQ_OR_RETURN(permute.size(), ndim);
  CheckIsPerm(permute);
  DimVector target_dims(ndim);

  StrideVector stride_vec(ndim);
  for (int i = 0; i < ndim; ++i) {
    target_dims[i] = shape->At(permute.at(i));
    stride_vec[i] = strides->At(permute.at(i));
  }

  auto output = JUST(BasicView(input, Shape(target_dims), Stride(stride_vec), storage_offset));
  if (input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              std::vector<int32_t> grad_perm;
              grad_perm.resize(ndim);
              for (int i = 0; i < ndim; ++i) { grad_perm.at(permute.at(i)) = i; }
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) = JUST(functional::Transpose(out_grads.at(0), grad_perm));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::transpose_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> UnfoldTensor(const std::shared_ptr<Tensor>& input, const MutableAttrMap& attrs) {
  const auto& shape = input->shape();
  const auto& stride = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());

  AttrMap attr_map(attrs);
  const int32_t dimension = JUST(attr_map.GetAttr<int32_t>("dimension"));
  const int32_t size = JUST(attr_map.GetAttr<int32_t>("size"));
  const int32_t step = JUST(attr_map.GetAttr<int32_t>("step"));
  CHECK_GE_OR_RETURN(dimension, 0);
  CHECK_LE_OR_RETURN(dimension, ndim - 1);

  const int32_t max_size = ndim == 0 ? 1 : shape->At(dimension);
  CHECK_GT_OR_RETURN(size, 0);
  CHECK_LE_OR_RETURN(size, max_size);
  CHECK_GT_OR_RETURN(step, 0);

  DimVector out_shape(ndim + 1);
  StrideVector out_stride(ndim + 1);
  out_shape[ndim] = size;
  out_stride[ndim] = ndim == 0 ? 1 : stride->At(dimension);
  for (int64_t d = 0; d < ndim; ++d) {
    const int64_t in_size_at_d = shape->At(d);
    if (d == dimension) {
      out_shape.at(d) = (in_size_at_d - size) / step + 1;
      out_stride.at(d) = step * stride->At(d);
    } else {
      out_shape.at(d) = in_size_at_d;
      out_stride.at(d) = stride->At(d);
    }
  }
  auto output = JUST(BasicView(input, Shape(out_shape), Stride(out_stride), storage_offset));

  if (input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) =
                  JUST(functional::UnfoldTensorGrad(out_grads.at(0), input, dimension, size, step));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::unfold_tensor_backward",
                                                            backward_fn, {input}, &outputs));
  }

  return output;
}

Maybe<Tensor> Diagonal(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& dx,
                       const int32_t offset, const int32_t dim1, const int32_t dim2) {
  const auto& shape = input->shape();
  const auto& stride = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();
  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());

  // infer output storage_offset
  int64_t diag_size;
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(shape->At(dim1), shape->At(dim2) - offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(shape->At(dim1) + offset, shape->At(dim2)), 0);
  }
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * stride->At(dim2);
  } else {
    storage_offset -= offset * stride->At(dim1);
  }

  CHECK_GE_OR_RETURN(ndim, 2);
  // infer output shape and stride
  DimVector out_shape(shape->dim_vec());
  StrideVector out_stride(stride->StrideVec());
  out_shape.erase(out_shape.begin() + std::max(dim1, dim2));
  out_stride.erase(out_stride.begin() + std::max(dim1, dim2));
  out_shape.erase(out_shape.begin() + std::min(dim1, dim2));
  out_stride.erase(out_stride.begin() + std::min(dim1, dim2));
  out_shape.emplace_back(diag_size);
  out_stride.emplace_back(stride->At(dim1) + stride->At(dim2));

  // generate view tensor
  auto output = JUST(BasicView(input, Shape(out_shape), Stride(out_stride), storage_offset));
  // autograd
  if (input->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) = JUST(functional::DiagonalGrad(out_grads.at(0), dx, offset));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::diagonal_backward", backward_fn,
                                                            {dx}, &outputs));
  }

  return output;
}

}  // namespace view
}  // namespace one
}  // namespace oneflow
