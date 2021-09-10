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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/functional/functional.h"

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

Maybe<Tensor> Slice(const std::shared_ptr<Tensor>& tensor, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& steps) {
  if (!(tensor->is_eager() && tensor->is_local())) {
    return Error::RuntimeError() << "view::Slice(): input should be eager local tensor, but is "
                                 << (tensor->is_lazy() ? "lazy" : "consistent");
  }

  // const auto& callback =
  //     std::make_shared<std::function<void(uint64_t)>>([](uint64_t of_blob_ptr) {});
  // CHECK_JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void>
  // {
  //   return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
  //     return builder->SyncAccessBlobByCallback(JUST(tensor->AsMirroredTensor()), sc, callback,
  //                                              "const");
  //   });
  // }));

  const auto& shape = tensor->shape();
  const auto& strides = JUST(tensor->stride());
  int size = starts.size();
  if (size != shape->NumAxes()) {
    return Error::RuntimeError() << "view::Slice(): starts size is expected " << shape->NumAxes()
                                 << ", but got " << size;
  }
  if (ends.size() != size || steps.size() != size) {
    return Error::RuntimeError() << "view::Slice(): " << (ends.size() != size ? "ends" : "steps")
                                 << " size is not equal to start.";
  }
  DimVector target_dims(size);
  StrideVector target_strides(size);
  int64_t storage_offset = JUST(tensor->storage_offset());
  for (int i = 0; i < size; ++i) {
    int64_t step = std::min(steps.at(i), shape->At(i));
    if (step < 0) { return Error::RuntimeError() << "Step must be greater than zero."; }
    int64_t start = std::min(starts.at(i), shape->At(i));
    int64_t end = std::min(ends.at(i), shape->At(i));
    if (start < 0) { start += shape->At(i); }
    if (start < 0) start = 0;
    if (end < 0) { end += shape->At(i); }
    if (end < start) end = start;
    int64_t length = start == end ? 0 : (end - start + step - 1) / step;
    target_dims[i] = length;
    target_strides[i] = step * strides->At(i);
    storage_offset += start * strides->At(i);
  }
  // Slice 1-d tensor maybe generate 0-dim tensor.
  if (size == 1 && target_dims.at(0) == 1) { target_dims = DimVector{}; }
  auto tensor_meta = std::make_shared<MirroredTensorMeta>(
      std::make_shared<Shape>(target_dims), tensor->dtype()->data_type(), JUST(tensor->device()),
      std::make_shared<Stride>(target_strides), storage_offset);

  JUST(tensor->has_eager_blob_object());
  const auto& blob_object = JUST(tensor->eager_blob_object());

  auto tensor_impl = std::make_shared<EagerMirroredTensorImpl>(
      tensor_meta, JUST(tensor->tensor_storage()), tensor->requires_grad(), tensor->is_leaf());
  tensor_impl->InitEagerBlobObject(JUST(blob_object->compute_local_dep_object()));
  JUST(JUST(tensor_impl->eager_blob_object())->TryInitBlob());
  JUST(tensor_impl->eager_blob_object())->set_is_shape_synced(true);
  std::shared_ptr<Tensor> output(new MirroredTensor(tensor_impl));
  if (tensor->requires_grad()) {
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              in_grads->at(0) =
                  JUST(functional::SliceGrad(out_grads.at(0), tensor, starts, ends, steps));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("vew::slice_backward", backward_fn,
                                                            {tensor}, &outputs));
  }
  return output;
}

Maybe<Tensor> Transpose(const std::shared_ptr<Tensor>& tensor,
                        const std::vector<int32_t>& permute) {
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace view

}  // namespace one
}  // namespace oneflow
