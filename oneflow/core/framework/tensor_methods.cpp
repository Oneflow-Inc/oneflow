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

Maybe<bool> IsContiguous(const std::shared_ptr<Tensor>& tensor) {
  const Shape& shape = *tensor->shape();
  const Stride& stride = *JUST(tensor->stride());
  int64_t dim = shape.NumAxes();
  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; --i) {
    // Contiguous by default when any dim is equal to zero
    // https://stackoverflow.com/questions/31681324/identify-contiguous-segments-of-a-non-contiguous-numpy-array
    if (shape.At(i) == 0) { return true; }
    if (contig_if_nonempty && shape.At(i) != 1) {
      if (stride.At(i) != expected_stride) { contig_if_nonempty = false; }
      expected_stride *= shape.At(i);
    }
  }
  return contig_if_nonempty;
}

namespace view {

bool IsViewApplicable(const std::shared_ptr<Tensor>& input) {
  // NOTE: only eager local tensor support view for now
  // elem_cnt() > 1  used to excluding 0 shape tensor
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
  Stride target_stride(target_shape);
  return Reshape(input, target_shape, target_stride);
}

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                      const Stride& target_stride) {
  // TODO:(zhaoluyang) check input tensor is contiguous
  CHECK_OR_RETURN(IsViewApplicable(input))
      << Error::RuntimeError()
      << "view::Reshape(): input should be eager local tensor with element count >=1 , but got "
      << (input->is_lazy() ? "lazy tensor" : "consistent tensor")
      << " with shape: " << input->shape()->ToString() << "; element count: " << input->nelement();

  int64_t storage_offset = JUST(JUST(input->AsMirroredTensor())->storage_offset());
  std::shared_ptr<Tensor> output =
      JUST(BasicView(input, target_shape, target_stride, storage_offset));

  if (autograd::GradMode::is_enabled() && input->requires_grad()) {
    Shape input_shape(input->shape()->dim_vec());
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              *JUST(oneflow::VectorAt(in_grads, 0)) =
                  JUST(functional::Reshape(JUST(oneflow::VectorAt(out_grads, 0)), input_shape));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::reshape_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> Slice(const std::shared_ptr<Tensor>& input, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& steps) {
  CHECK_OR_RETURN(IsViewApplicable(input))
      << Error::RuntimeError() << "view::Slice(): input should be eager local tensor, but is "
      << (input->is_lazy() ? "lazy tensor" : "consistent tensor")
      << " with shape: " << input->shape()->ToString() << "; element count: " << input->nelement();
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

Maybe<Tensor> Unsqueeze(const std::shared_ptr<Tensor>& input, const int32_t& expand_dim) {
  CHECK_OR_RETURN(IsViewApplicable(input))
      << Error::RuntimeError() << "view::Unsqueeze(): input should be eager local tensor, but got "
      << (input->is_lazy() ? "lazy tensor" : "consistent tensor")
      << " with shape: " << input->shape()->ToString() << "; element count: " << input->nelement();

  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const auto& ndim = shape->NumAxes();

  DimVector target_dim_vec(ndim + 1);
  StrideVector target_stride_vec(ndim + 1);

  {
    int cnt = 0;
    for (int i = 0; i < ndim; i++) {
      if (i == expand_dim) { cnt++; }
      target_dim_vec[cnt] = shape->At(i);
      target_stride_vec[cnt] = strides->At(i);
      cnt++;
    }
    target_dim_vec[expand_dim] = 1;
    target_stride_vec[expand_dim] = strides->At(expand_dim);
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
              *JUST(oneflow::VectorAt(in_grads, 0)) =
                  JUST(functional::Reshape(JUST(oneflow::VectorAt(out_grads, 0)), *shape));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::unsqueeze_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

Maybe<Tensor> Squeeze(const std::shared_ptr<Tensor>& input,
                      const std::vector<int32_t>& squeeze_dims) {
  CHECK_OR_RETURN(IsViewApplicable(input))
      << Error::RuntimeError() << "view::Squeeze(): input should be eager local tensor, but got "
      << (input->is_lazy() ? "lazy tensor" : "consistent tensor")
      << " with shape: " << input->shape()->ToString() << "; element count: " << input->nelement();

  const auto& shape = input->shape();
  const auto& strides = JUST(input->stride());
  const int64_t ndim = shape->NumAxes();

  const int target_ndim = ndim - squeeze_dims.size();
  DimVector target_dim_vec(target_ndim);
  StrideVector target_stride_vec(target_ndim);

  {
    int cnt = 0;
    for (int i = 0; i < ndim; i++) {
      if (find(squeeze_dims.begin(), squeeze_dims.end(), i) == squeeze_dims.end()) {
        target_dim_vec[cnt] = shape->At(i);
        target_stride_vec[cnt] = strides->At(i);
        cnt++;
      }
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
              *JUST(oneflow::VectorAt(in_grads, 0)) = JUST(functional::Reshape(
                  JUST(oneflow::VectorAt(out_grads, 0)), Shape(input->shape()->dim_vec())));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::squeeze_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

}  // namespace view
}  // namespace one
}  // namespace oneflow
