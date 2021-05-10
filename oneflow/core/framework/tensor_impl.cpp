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
#include "oneflow/api/foreign_lock_helper.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/vm/vm_util.h"

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
    const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
    const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf)
    : MirroredTensorImpl(device, requires_grad, is_leaf), eager_blob_object_(eager_blob_object) {
  dtype_ = CHECK_JUST(DType::GetDTypeByDataType(eager_blob_object->blob_desc().data_type()));
  tensor_storage_ = std::make_shared<TensorStorage>(eager_blob_object->tensor_buffer());
  const auto& parallel_desc = this->parallel_desc();
  tensor_storage_->set_releaser_hook(
      [eager_blob_object, parallel_desc](const std::shared_ptr<vm::TensorBuffer>&) {
        PhysicalRun([&](InstructionsBuilder* builder) {
          builder->ReleaseTensor(eager_blob_object, parallel_desc);
        });
      });
}

Maybe<VmLocalDepObject> EagerMirroredTensorImpl::infer_local_dep_object() const {
  return eager_blob_object_->infer_local_dep_object();
}

Maybe<VmLocalDepObject> EagerMirroredTensorImpl::compute_local_dep_object() const {
  return eager_blob_object_->compute_local_dep_object();
}

const std::shared_ptr<const Shape>& EagerMirroredTensorImpl::shape() const {
  if (eager_blob_object_->is_shape_synced()) { return eager_blob_object_->blob_desc().shape_ptr(); }

  const std::shared_ptr<const Shape>* result = nullptr;
  Global<ForeignLockHelper>::Get()->WithScopedRelease([this, &result]() {
    BlockingCounter bc(1);
    auto callback = [&bc, &result](const std::shared_ptr<const Shape>& shape) -> void {
      result = &shape;
      bc.Decrease();
    };
    auto build_instruction = [&](InstructionsBuilder* builder) -> Maybe<void> {
      JUST(builder->ReadTensorShapeByCallback(JUST(eager_blob_object()), callback));
      return Maybe<void>::Ok();
    };
    CHECK_JUST(PhysicalRun(build_instruction));
    bc.WaitUntilCntEqualZero();
  });
  eager_blob_object_->set_is_shape_synced(true);
  return *result;
}

}  // namespace one
}  // namespace oneflow
