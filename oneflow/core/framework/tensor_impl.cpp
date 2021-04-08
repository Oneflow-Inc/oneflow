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
#include <type_traits>
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace one {

Maybe<void> TensorImpl::SyncBlobObject2Attributes(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  set_shape(blob_object->op_arg_blob_attr()->shape());
  DataType data_type = static_cast<DataType>(blob_object->op_arg_blob_attr()->get_dtype());
  const std::shared_ptr<DType>& dtype = JUST(DType::GetDTypeByDataType(data_type));
  set_dtype(dtype);
  return set_parallel_desc(blob_object->op_arg_parallel_attr()->parallel_desc_symbol());
}

Maybe<void> MirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) {
  device_ = device;
  parallel_desc_ = JUST(Device::MakeParallelDescByDevice(*device));
  return Maybe<void>::Ok();
}

Maybe<void> MirroredTensorImpl::set_parallel_desc(
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  parallel_desc_ = parallel_desc;
  device_ = JUST(Device::MakeDeviceByParallelDesc(*parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> ConsistentTensorImpl::set_parallel_desc(
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  parallel_desc_ = parallel_desc;
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  return SyncBlobObject2Attributes(blob_object);
}

Maybe<void> EagerConsistentTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  return SyncBlobObject2Attributes(blob_object);
}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(const std::shared_ptr<const Shape>& shape,
                                                 const std::shared_ptr<const DType>& dtype,
                                                 const std::shared_ptr<const Device>& device,
                                                 bool requires_grad, bool is_leaf, bool retain_grad)
    : EagerMirroredTensorImpl(shape, dtype, device, std::make_shared<TensorStorage>(),
                              requires_grad, is_leaf, retain_grad) {}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(
    const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
    const std::shared_ptr<const Device>& device,
    const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad, bool is_leaf,
    bool retain_grad)
    : MirroredTensorImpl(device, requires_grad, is_leaf, retain_grad),
      shape_(shape),
      dtype_(dtype),
      tensor_storage_(tensor_storage),
      vm_local_dep_object_(parallel_desc()) {}

Maybe<void> EagerMirroredTensorImpl::InitEagerBlobObject(
    const std::shared_ptr<MemoryCase>& mem_case) {
  CHECK_OR_RETURN(!static_cast<bool>(eager_blob_object_));
  eager_blob_object_.reset(
      new eager::EagerBlobObject(mem_case, std::const_pointer_cast<Shape>(shape_),
                                 dtype_->data_type(), tensor_storage_->buffer()));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
