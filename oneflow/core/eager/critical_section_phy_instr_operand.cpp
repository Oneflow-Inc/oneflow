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
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {
namespace vm {

CriticalSectionPhyInstrOperand::CriticalSectionPhyInstrOperand(int64_t ref_cnt)
    : critical_section_ready_ref_cnt_(std::make_shared<std::atomic<int64_t>>(1)),
      consumer_ref_cnt_(std::make_shared<std::atomic<int64_t>>(ref_cnt)) {}

TensorCriticalSectionPhyInstrOperand::TensorCriticalSectionPhyInstrOperand(
    const one::EagerBlobObjectListPtr& eager_blob_objects,
    const HashMap<std::string, int64_t>& op_name2index, const std::shared_ptr<NNGraphIf>& nn_graph)
    : CriticalSectionPhyInstrOperand(eager_blob_objects->size()),
      eager_blob_objects_(eager_blob_objects),
      nn_graph_(nn_graph),
      op_name2index_(op_name2index) {}

void TensorCriticalSectionPhyInstrOperand::ConsumerFetchBlobAndDecreaseRefCnt(
    const std::string& op_name, const std::function<void(Blob*)>& Callback) const {
  {
    const auto& iter = op_name2index_.find(op_name);
    CHECK(iter != op_name2index_.end());
    const auto& eager_blob_object = eager_blob_objects_->at(iter->second);
    CHECK(static_cast<bool>(eager_blob_object));
    Callback(eager_blob_object->mut_blob());
  }
  CHECK_GE(*consumer_ref_cnt_, 0);
  --*consumer_ref_cnt_;
}

void TensorCriticalSectionPhyInstrOperand::ForEachMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& eager_blob_object : *eager_blob_objects_) {
    DoEach(nullptr,
           CHECK_JUST(eager_blob_object->compute_local_dep_object())->mut_mirrored_object());
  }
}

namespace {

Maybe<LocalDepObject*> RawCriticalSectionLocalDepObject() {
  return JUST(Device::New("critical_section"))->mut_schedule_local_dep_object();
}

constexpr auto* CriticalSectionLocalDepObject =
    DECORATE(&RawCriticalSectionLocalDepObject, ThreadLocal);

}  // namespace

void InputCriticalSectionPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr, CHECK_JUST(CriticalSectionLocalDepObject())->mut_mirrored_object());
}

void OutputCriticalSectionPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr, CHECK_JUST(CriticalSectionLocalDepObject())->mut_mirrored_object());
}

void ParameterCriticalSectionPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr, CHECK_JUST(CriticalSectionLocalDepObject())->mut_mirrored_object());
  ForEachMirroredObject(DoEach);
}

namespace {

Maybe<LocalDepObject*> RawGetEagerNcclLocalDepObject(const std::string& type) {
  const auto& device = JUST(Device::New(type));
  const auto& local_dep_object = device->mut_transport_local_dep_object();
  CHECK_OR_RETURN(local_dep_object.has_value());
  return JUST(local_dep_object);
}

static constexpr auto* GetEagerNcclLocalDepObject =
    DECORATE(&RawGetEagerNcclLocalDepObject, ThreadLocalCopiable);

}  // namespace

void NcclCriticalSectionPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr, CHECK_JUST(CriticalSectionLocalDepObject())->mut_mirrored_object());
#ifdef WITH_CUDA
  auto* sync_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("sync_launched_nccl"));
  auto* async_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("async_launched_nccl"));
  CHECK_EQ(sync_launched_nccl, async_launched_nccl);
  DoEach(nullptr, async_launched_nccl->mut_mirrored_object());
#endif  // WITH_CUDA
}

}  // namespace vm
}  // namespace oneflow
