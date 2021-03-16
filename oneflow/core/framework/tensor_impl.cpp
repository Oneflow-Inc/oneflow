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
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace one {

Maybe<void> TensorImpl::SyncBlobObject2Attributes(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  set_shape(blob_object->op_arg_blob_attr()->shape());
  DataType data_type = static_cast<DataType>(blob_object->op_arg_blob_attr()->get_dtype());
  const std::shared_ptr<DType>& dtype = JUST(DType::GetDTypeByDataType(data_type));
  set_dtype(dtype);
  return Maybe<void>::Ok();
}

void MirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) {
  device_ = device;
  parallel_desc_ = Device::MakeParallelDescByDevice(*device);
}

Maybe<void> MirroredTensorImpl::set_parallel_desc(
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  parallel_desc_ = parallel_desc;
  device_ = JUST(Device::MakeDeviceByParallelDesc(*parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  JUST(SyncBlobObject2Attributes(blob_object));
  JUST(set_parallel_desc(blob_object->op_arg_parallel_attr()->parallel_desc_symbol()));
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  JUST(SyncBlobObject2Attributes(blob_object));
  set_parallel_desc(blob_object->op_arg_parallel_attr()->parallel_desc_symbol());
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
