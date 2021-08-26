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

Maybe<MirroredTensor> ShallowCopy(const std::shared_ptr<MirroredTensor>& tensor,
                                  const std::shared_ptr<const Shape>& new_shape,
                                  const std::shared_ptr<const Stride>& new_stride,
                                  int64_t new_storage_offset, DataType new_dtype) {
  const auto& meta = static_cast<const MirroredTensorMeta&>(tensor->tensor_meta());
  const auto& shape = new_shape ? new_shape : std::make_shared<const Shape>(meta.shape());
  const auto& dtype = new_dtype != kInvalidDataType ? new_dtype : meta.dtype();

  auto new_meta = std::make_shared<MirroredTensorMeta>(
      shape, dtype, meta.device(),
      new_stride ? new_stride : std::make_shared<const Stride>(meta.stride()),
      new_storage_offset != -1 ? new_storage_offset : meta.storage_offset());

  const auto& impl = tensor->mut_impl();
  const auto& blob_obj = JUST(impl->eager_blob_object());

  auto new_blob_obj = std::shared_ptr<vm::EagerBlobObject>(new vm::EagerBlobObject(
      std::make_shared<MemoryCase>(blob_obj->mem_case()), std::const_pointer_cast<Shape>(shape),
      dtype, blob_obj->tensor_buffer(), blob_obj->compute_local_dep_object_));

  JUST(new_blob_obj->InitBlob());
  new_blob_obj->blob_body_bytes_ = blob_obj->blob_body_bytes_;
  new_blob_obj->mut_blob()->reset_dptr(blob_obj->mut_blob()->mut_dptr<char>());

  auto new_impl = std::make_shared<EagerMirroredTensorImpl>(new_meta, JUST(impl->tensor_storage()),
                                                            new_blob_obj, tensor->requires_grad(),
                                                            tensor->is_leaf());

  return std::make_shared<MirroredTensor>(new_impl);
}

}  // namespace one
}  // namespace oneflow
