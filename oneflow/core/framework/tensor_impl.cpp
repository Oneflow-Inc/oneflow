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

/*static*/ Maybe<EagerConsistentTensorImpl> EagerConsistentTensorImpl::New(
    const std::shared_ptr<EagerMirroredTensorImpl>& cur_rank_phy_tensor_impl,
    const std::shared_ptr<const cfg::ParallelDistribution>& parallel_distribution,
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  {
    int64_t machine_id = 0;
    int64_t device_id = 0;
    GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
    const auto& device = JUST(Device::ThreadLocalGetOrNew(parallel_desc->device_tag(), device_id));
    CHECK_OR_RETURN(*device == *cur_rank_phy_tensor_impl->device())
        << "only LocalTensors on current rank Device can be casted to ConsistentTensor";
  }
  const auto& shape = JUST(
      GetLogicalShape(*cur_rank_phy_tensor_impl->shape(), *parallel_distribution, *parallel_desc));
  const auto& dtype = cur_rank_phy_tensor_impl->dtype();
  const auto& autograd_meta = cur_rank_phy_tensor_impl->mut_autograd_meta();
  return std::shared_ptr<EagerConsistentTensorImpl>(new EagerConsistentTensorImpl(
      shape, dtype, parallel_distribution, parallel_desc, cur_rank_phy_tensor_impl, autograd_meta));
}

/*static*/ Maybe<EagerConsistentTensorImpl> EagerConsistentTensorImpl::New(
    const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
    const std::shared_ptr<const cfg::ParallelDistribution>& parallel_distribution,
    const std::shared_ptr<const ParallelDesc>& parallel_desc, bool requires_grad, bool is_leaf) {
  const auto& autograd_meta = NewAutogradMeta(requires_grad, is_leaf);
  std::shared_ptr<EagerMirroredTensorImpl> cur_rank_phy_tensor_impl;
  {
    int64_t machine_id = 0;
    int64_t device_id = 0;
    GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
    int64_t parallel_id = JUST(parallel_desc->ParallelId4MachineDeviceId(machine_id, device_id));
    const auto& cur_rank_phy_shape =
        JUST(GetPhysicalShape(*shape, *parallel_distribution, *parallel_desc, parallel_id));
    const auto& device = JUST(Device::ThreadLocalGetOrNew(parallel_desc->device_tag(), device_id));
    const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
        device->mem_case(), cur_rank_phy_shape, dtype->data_type(),
        std::make_shared<vm::TensorBuffer>(), device->parallel_desc_ptr());
    cur_rank_phy_tensor_impl.reset(
        new EagerMirroredTensorImpl(eager_blob_object, device, autograd_meta));
    cur_rank_phy_tensor_impl->set_shape(cur_rank_phy_shape);
    cur_rank_phy_tensor_impl->set_dtype(dtype);
  }
  return std::shared_ptr<EagerConsistentTensorImpl>(new EagerConsistentTensorImpl(
      shape, dtype, parallel_distribution, parallel_desc, cur_rank_phy_tensor_impl, autograd_meta));
}

}  // namespace one
}  // namespace oneflow
