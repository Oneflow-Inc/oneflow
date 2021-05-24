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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace one {

Maybe<void> MirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) {
  device_ = device;
  return Maybe<void>::Ok();
}

EagerMirroredTensorImpl::~EagerMirroredTensorImpl() {}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(
    const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
    const std::shared_ptr<const Device>& device, const std::shared_ptr<AutogradMeta>& autograd_meta)
    : MirroredTensorImpl(device, autograd_meta), eager_blob_object_(eager_blob_object) {
  Init();
}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(
    const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
    const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf)
    : MirroredTensorImpl(device, NewAutogradMeta(requires_grad, is_leaf)),
      eager_blob_object_(eager_blob_object) {
  Init();
}

void EagerMirroredTensorImpl::Init() {
  const auto& eager_blob_object = eager_blob_object_;
  dtype_ = CHECK_JUST(DType::GetDTypeByDataType(eager_blob_object->blob_desc().data_type()));
  tensor_storage_ = std::make_shared<TensorStorage>(eager_blob_object->tensor_buffer());
  const auto& parallel_desc = this->device()->parallel_desc_ptr();
  tensor_storage_->set_releaser_hook(
      [eager_blob_object, parallel_desc](const std::shared_ptr<vm::TensorBuffer>&) {
        PhysicalRun([&](InstructionsBuilder* builder) {
          builder->ReleaseTensor(eager_blob_object, parallel_desc);
        });
      });
}

Maybe<VmLocalDepObject> EagerMirroredTensorImpl::compute_local_dep_object() const {
  return eager_blob_object_->compute_local_dep_object();
}

const std::shared_ptr<const Shape>& EagerMirroredTensorImpl::shape() const {
  if (eager_blob_object_->is_shape_synced()) { return eager_blob_object_->blob_desc().shape_ptr(); }

  std::atomic<bool> synced(false);

  PhysicalRun([&](InstructionsBuilder* builder) {
    builder->AccessBlobByCallback(
        this, [&synced](uint64_t) { synced = true; }, "const");
  });

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() {
    // spin wait
    while (!synced) {}
  });

  eager_blob_object_->set_is_shape_synced(true);
  return eager_blob_object_->blob_desc().shape_ptr();
}

bool ConsistentTensorMeta::operator==(const ConsistentTensorMeta& other) const {
  return *this->shape() == *other->shape()
    && *this->dtype() == *other.dtype()
    && *this->parallel_distribution() == *other.parallel_distribution()
    && *this->parallel_desc() == *other.parallel_desc();
}

size_t ConsistentTensorMeta::CalcHashValue() const {
  return std::hash<Shape>()(*shape())
    ^ std::hash<DType>()(*dtype())
    ^ std::hash<cfg::ParallelDistribution>()(*parallel_distribution)
    ^ std::hash<ParallelDesc>()(*parallel_desc);
}

/*static*/ Maybe<EagerConsistentTensorImpl> EagerConsistentTensorImpl::New(
    const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor,
    const std::shared_ptr<const cfg::ParallelDistribution>& parallel_distribution,
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  CHECK_OR_RETURN(!cur_rank_phy_tensor.is_lazy());
  {
    int64_t machine_id = 0;
    int64_t device_id = 0;
    GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
    const auto& device = JUST(Device::ThreadLocalGetOrNew(parallel_desc->device_tag(), device_id));
    const auto& device = JUST(cur_rank_phy_tensor->device());
    CHECK_OR_RETURN(*device == *device)
        << "only LocalTensors on current rank Device can be casted to ConsistentTensor";
  }
  const auto& shape = JUST(
      GetLogicalShape(*cur_rank_phy_tensor->shape(), *parallel_distribution, *parallel_desc));
  const auto& dtype = cur_rank_phy_tensor->dtype();
  Symbol<ConsistentTensorMeta> consistent_tensor_meta(
      ConsistentTensorMeta(shape, dtype, parallel_distribution, parallel_desc));
  return std::shared_ptr<EagerConsistentTensorImpl>(
      new EagerConsistentTensorImpl(consistent_tensor_meta, cur_rank_phy_tensor));
}

/*static*/ Maybe<EagerConsistentTensorImpl> EagerConsistentTensorImpl::New(
    Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad, bool is_leaf) {
  std::shared_ptr<MirroredTensor> cur_rank_phy_tensor;
  {
    int64_t machine_id = 0;
    int64_t device_id = 0;
    GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
    int64_t parallel_id = JUST(parallel_desc->ParallelId4MachineDeviceId(machine_id, device_id));
    const auto& shape = consistent_tensor_meta.shape();
    const auto& parallel_distribution = consistent_tensor_meta.parallel_distribution();
    const auto& parallel_desc = consistent_tensor_meta.parallel_desc();
    const auto& cur_rank_phy_shape =
        JUST(GetPhysicalShape(*shape, *parallel_distribution, *parallel_desc, parallel_id));
    const auto& device = JUST(Device::ThreadLocalGetOrNew(parallel_desc->device_tag(), device_id));
    const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
        device->mem_case(), cur_rank_phy_shape, dtype->data_type(),
        std::make_shared<vm::TensorBuffer>(), device->parallel_desc_ptr());
    const auto& autograd_meta = NewAutogradMeta(requires_grad, is_leaf);
    const auto& cur_rank_phy_tensor_impl =
      std::make_shared<EagerMirroredTensorImpl>(eager_blob_object, device, autograd_meta);
    cur_rank_phy_tensor_impl->set_shape(cur_rank_phy_shape);
    cur_rank_phy_tensor_impl->set_dtype(dtype);
    cur_rank_phy_tensor.reset(new MirroredTensor(cur_rank_phy_tensor_impl));
  }
  return std::shared_ptr<EagerConsistentTensorImpl>(new EagerConsistentTensorImpl(
      consistent_tensor_meta, cur_rank_phy_tensor));
}

}  // namespace one
}  // namespace oneflow
