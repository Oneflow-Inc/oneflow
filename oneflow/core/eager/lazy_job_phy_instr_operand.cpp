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
#include "oneflow/core/eager/lazy_job_phy_instr_operand.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace vm {

namespace {

#ifdef WITH_CUDA
Maybe<LocalDepObject*> RawGetEagerNcclLocalDepObject(const std::string& type) {
  const auto& device = JUST(Device::New(type));
  const auto& local_dep_object = device->mut_transport_local_dep_object();
  CHECK_OR_RETURN(local_dep_object.has_value());
  return JUST(local_dep_object);
}

static constexpr auto* GetEagerNcclLocalDepObject =
    DECORATE(&RawGetEagerNcclLocalDepObject, ThreadLocalCopiable);
#endif  // WITH_CUDA

}  // namespace

void LaunchLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  DoEach(inputs_local_dep_object_->mut_mirrored_object());
  DoEach(outputs_local_dep_object_->mut_mirrored_object());

  for (const auto& eager_blob_object : *param_blob_objects_) {
    DoEach(CHECK_JUST(eager_blob_object->compute_local_dep_object())->mut_mirrored_object());
  }

#ifdef WITH_CUDA
  auto* sync_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("sync_launched_nccl"));
  auto* async_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("async_launched_nccl"));
  CHECK_EQ(sync_launched_nccl, async_launched_nccl);
  DoEach(async_launched_nccl->mut_mirrored_object());
#endif  // WITH_CUDA
}

void LaunchLazyJobPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  DoEach(inputs_local_dep_object_->mut_mirrored_object());
}

void LaunchLazyJobPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  DoEach(outputs_local_dep_object_->mut_mirrored_object());
}

Maybe<SharedEventRecord> LaunchLazyJobPhyInstrOperand::EndEventRecord4OpName(
    const std::string& op_name) const {
  return JUST(MapAt(*op_name2end_event_record_, op_name));
}

}  // namespace vm
}  // namespace oneflow
