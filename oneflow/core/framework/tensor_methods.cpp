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

Maybe<void> SyncAccessTensorWithTimeOut(
    const std::shared_ptr<Tensor>& tensor,
    const std::shared_ptr<std::function<void(uint64_t)>>& callback, const std::string& modifier) {
  return SpinCounter::SpinWait(3, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    printf("\n SyncAccessTensorWithTimeOut >>>> SpinCounter");
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      printf("\n SyncAccessTensorWithTimeOut >>>> PhysicalRun");
      return builder->SyncAccessBlobByCallback(JUST(tensor->AsMirroredTensor()), sc, callback,
                                               modifier);
    });
  });
}


Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                         const Stride& target_strides, int64_t storage_offset) {

  const void* input_ptr = nullptr;
  const auto& callback =
      std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
        printf("\n BasicView >>>>> SyncAccessTensorWithTimeOut >>>> callback input");
        // auto* eager_blob = reinterpret_cast<vm::EagerBlobObject*>(of_blob_ptr);
        auto* eager_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        input_ptr = eager_blob->blob().dptr();
      });
  JUST(SyncAccessTensorWithTimeOut(input, callback, "const")); // const or mut ? 

  storage_offset += JUST(input->storage_offset());
  // TODO(): Check shape compatible.
  auto tensor_meta = std::make_shared<MirroredTensorMeta>(
      std::make_shared<Shape>(target_shape), input->dtype()->data_type(), JUST(input->device()),
      std::make_shared<Stride>(target_strides), storage_offset);

  JUST(input->has_eager_blob_object());
  const auto& blob_object = JUST(input->eager_blob_object());

  auto tensor_impl = std::make_shared<EagerMirroredTensorImpl>(
      tensor_meta, JUST(input->tensor_storage()), input->requires_grad(),
      /*is_leaf=*/!input->requires_grad());
  tensor_impl->InitEagerBlobObject(JUST(blob_object->compute_local_dep_object()));

  // const auto& dep_object = JUST(GetLocalDepObjectFromDevicePool(JUST(Device::New("cpu"))));
  // tensor_impl->InitEagerBlobObject(dep_object);

  JUST(JUST(tensor_impl->eager_blob_object())->TryInitBlob());
  JUST(tensor_impl->eager_blob_object())->set_is_shape_synced(true);
  JUST(tensor_impl->eager_blob_object())->set_last_used_device(JUST(input->device()));
  std::shared_ptr<Tensor> output(new MirroredTensor(tensor_impl));
  printf("\n callback 1 finish");

  {
    const auto& callback =
        std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
          printf("\n BasicView >>>>> SyncAccessTensorWithTimeOut >>>> callback output");
          // auto* eager_blob = reinterpret_cast<vm::EagerBlobObject*>(of_blob_ptr);
          auto* eager_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
          if (!(eager_blob->blob().dptr())) {
            printf("\n eager_blob->blob().dptr() >>>> not exist");
            int64_t storage_offset_bytes = storage_offset * GetSizeOfDataType(eager_blob->blob().data_type());
            eager_blob->mut_blob()->reset_dptr((char*)input_ptr + storage_offset_bytes);
          }else{
            printf("\n eager_blob->blob().dptr() >>>> exist");
          }
        });
    JUST(SyncAccessTensorWithTimeOut(output, callback, "mut")); // const or mut ? 
  }
  return output;
}


Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& shape) {
  if (!(input->is_eager() && input->is_local())) {
    return Error::RuntimeError() << "view::Reshape(): input should be eager local tensor, but is "
                                 << (input->is_lazy() ? "lazy" : "consistent");
  }
  int need_infer_axis = -1;
  size_t count = 1;
  for (int i = 0; i < shape.NumAxes(); ++i) {
    if (shape.At(i) == -1) {
      CHECK_EQ_OR_RETURN(need_infer_axis, -1)
          << "Shape " << shape.ToString() << " has more than 1 axis that needs to be infered.";
      need_infer_axis = i;
    } else {
      count *= shape.At(i);
    }
  }

  std::shared_ptr<Tensor> output;
  size_t x_count = input->shape()->Count(0);
  if (need_infer_axis == -1) {
    CHECK_EQ_OR_RETURN(shape.Count(0), x_count);
    output = JUST(BasicView(input, shape, Stride(shape), 0));
  } else {
    Shape infered_shape = shape;
    infered_shape.Set(need_infer_axis, x_count / count);
    CHECK_EQ_OR_RETURN(infered_shape.Count(0), x_count)
        << "Shape " << shape.ToString() << " is invalid for input of shape "
        << input->shape()->ToString();
    output = JUST(BasicView(input, infered_shape, Stride(infered_shape), 0));
  }

  if (input->requires_grad()) {
    Shape input_shape(input->shape()->dim_vec());
    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              CHECK_EQ_OR_RETURN(out_grads.size(), 1);
              in_grads->resize(1);
              // in_grads->at(0) = JUST(functional::ReshapeLike(out_grads.at(0), input));
              in_grads->at(0) = JUST(functional::Reshape(out_grads.at(0), input_shape));
              return Maybe<void>::Ok();
            });
    TensorTuple outputs{output};
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr("view::reshape_backward", backward_fn,
                                                            {input}, &outputs));
  }
  return output;
}

}  // namespace view
}  // namespace one
}  // namespace oneflow
