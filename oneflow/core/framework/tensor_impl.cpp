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
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/register/ofblob.h"

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
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
}

EagerMirroredTensorImpl::~EagerMirroredTensorImpl() {}

Maybe<void> EagerConsistentTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  return SyncBlobObject2Attributes(blob_object);
}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(
    const std::shared_ptr<eager::EagerBlobObject> eager_blob_object,
    const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf, bool retain_grad)
    : MirroredTensorImpl(device, requires_grad, is_leaf, retain_grad),
      eager_blob_object_(eager_blob_object) {
  tensor_storage_ = std::make_shared<TensorStorage>(eager_blob_object->tensor_buffer());
}

Maybe<VmLocalDepObject> EagerMirroredTensorImpl::infer_local_dep_object() const {
  return eager_blob_object_->infer_local_dep_object();
}

Maybe<VmLocalDepObject> EagerMirroredTensorImpl::compute_local_dep_object() const {
  return eager_blob_object_->compute_local_dep_object();
}

const std::shared_ptr<const Shape> EagerMirroredTensorImpl::shape() const {
  BlockingCounter bc(1);
  std::shared_ptr<const Shape> result;
  auto callback = [&bc, &result](int64_t ofblob_ptr) -> void {
    OfBlob* ofblob = reinterpret_cast<OfBlob*>(ofblob_ptr);
    // TODO: use shared_ptr after rt_blob_desc is removed
    result = std::make_shared<const Shape>(ofblob->mut_blob()->blob_desc().body_shape());
    bc.Decrease();
  };
  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(parallel_desc(), eager_blob_object_, callback, "const"));
    return Maybe<void>::Ok();
  };
  CHECK_JUST(PhysicalRun(build_instruction));
  bc.WaitUntilCntEqualZero();
  return result;
}

const std::shared_ptr<const DType> EagerMirroredTensorImpl::dtype() const {
  BlockingCounter bc(1);
  std::shared_ptr<const DType> result;
  auto callback = [&bc, &result](int64_t ofblob_ptr) -> void {
    OfBlob* ofblob = reinterpret_cast<OfBlob*>(ofblob_ptr);
    result = CHECK_JUST(DType::GetDTypeByDataType(ofblob->mut_blob()->data_type()));
    bc.Decrease();
  };
  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(parallel_desc(), eager_blob_object_, callback, "const"));
    return Maybe<void>::Ok();
  };
  CHECK_JUST(PhysicalRun(build_instruction));
  bc.WaitUntilCntEqualZero();
  return result;
}

}  // namespace one
}  // namespace oneflow
