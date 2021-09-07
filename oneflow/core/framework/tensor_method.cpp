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

#include "oneflow/core/framework/tensor_method.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/eager/eager_blob_object.h"

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

Maybe<Tensor> TensorView(const std::shared_ptr<Tensor>& tensor,
                         const Optional<const Shape>& shape,
                         const Optional<const Stride>& stride,
                         int64_t storage_offset, const Optional<DataType>& dtype) {
  if (!(tensor->is_eager() && tensor->is_local())) {
    return Error::RuntimeError() << "TensorView(): input should be eager local tensor, but is "
                                 << tensor->is_lazy() ? "lazy" : "consistent";
  }
  CHECK_OR_RETURN(tensor->has_eager_blob_object());
  const auto& blob_object = JUST(tensor->eager_blob_object());

  auto to_shape = shape.value_or(tensor->shape());
  auto to_stride = stride.value_or(JUST(tensor->stride()));
  auto to_dtype = dtype.value_or(tensor->dtype());

  auto tensor_meta = std::make_shared<MirroredTensorMeta>(
      to_shape, to_dtype, JUST(tensor->device()),
      storage_offset != -1 ? storage_offset : JUST(tensor->storage_offset()));

  auto tensor_impl = std::make_shared<EagerMirroredTensorImpl>(tensor_meta, tensor->requires_grad(), tensor->is_leaf());
  tensor_impl->InitEagerBlobObject(JUST(blob_object->compute_local_dep_object()), blob_object->tensor_buffer());
  JUST(tensor_impl->eager_blob_object())->set_is_shape_synced(true);
  return std::make_shared<MirroredTensor>(tensor_impl);
}

}  // namespace one
}  // namespace oneflow
