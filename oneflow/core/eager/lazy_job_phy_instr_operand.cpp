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
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace vm {

CriticalSectionPhyInstrOperand::CriticalSectionPhyInstrOperand(
    const one::EagerBlobObjectListPtr& eager_blob_objects,
    const std::shared_ptr<NNGraphIf>& nn_graph)
      : eager_blob_objects_(eager_blob_objects), nn_graph_(nn_graph) {
  for (int i = 0; i < op_names().size(); ++i) {
    CHECK(op_name2index_.emplace(op_names().at(i), i).second);
    if (eager_blob_objects->at(i)) {
      CHECK(op_name2producer_notifier_.emplace(op_names().at(i),
            std::make_unique<Notifier>()).second);
      CHECK(op_name2consumer_notifier_.emplace(op_names().at(i),
            std::make_unique<Notifier>()).second);
    } else {
      // Do nothing because of no component of a consistent tensor found.
    }
  }
}

void CriticalSectionPhyInstrOperand::ProducerNotifyConsumerAndWaitConsumerAck() const {
  for (const auto& pair : op_name2producer_notifier_) {
    CHECK(pair.second->Notify() == kNotifierStatusSuccess);
  }
  for (const auto& pair : op_name2consumer_notifier_) {
    CHECK(pair.second->WaitAndClearNotifiedCnt() == kNotifierStatusSuccess);
  }
}

void CriticalSectionPhyInstrOperand::ConsumerFetchBlobAndNotifyProducerAck(const std::string& op_name, const std::function<void(Blob*)>& Callback) const {
  {
    const auto& iter = op_name2producer_notifier_.find(op_name);
    CHECK(iter != op_name2producer_notifier_.end());
    CHECK(iter->second->WaitAndClearNotifiedCnt() == kNotifierStatusSuccess);
  }
  {
    const auto& iter = op_name2index_.find(op_name);
    CHECK(iter != op_name2index_.end());
    const auto& eager_blob_object = eager_blob_objects_.at(iter->second);
    CHECK(static_cast<bool>(eager_blob_object));
    Callback(eager_blob_object->mut_blob());
  }
  {
    const auto& iter = op_name2consumer_notifier_.find(op_name);
    CHECK(iter != op_name2consumer_notifier_.end());
    CHECK(iter->second->Notify() == kNotifierStatusSuccess);
  }
}

void CriticalSectionPhyInstrOperand::ForEachMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach) const {
  for (const auto& eager_blob_object : *eager_blob_objects_) {
    if (eager_blob_object) {
      DoEach(nullptr, CHECK_JUST(eager_blob_object->compute_local_dep_object())->mut_mirrored_object());
    } else {
      // Do nothing because of no component of a consistent tensor found.
    }
  }
}

namespace {

Maybe<LocalDepObject*> RawGetEagerNcclLocalDepObject(const std::string& type) {
  const auto& device = JUST(Device::New(type));
  const auto& local_dep_object = device->mut_transport_local_dep_object();
  CHECK_OR_RETURN(local_dep_object.has_value());
  return JUST(local_dep_object.value());
}

}  // namespace

static constexpr auto* GetEagerNcclLocalDepObject =
    DECORATE(&RawGetEagerNcclLocalDepObject, ThreadLocalCopiable);

void LaunchLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
#ifdef WITH_CUDA
  auto* sync_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("sync_launched_nccl"));
  auto* async_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("async_launched_nccl"));
  CHECK_EQ(sync_launched_nccl, async_launched_nccl);
  DoEach(nullptr, async_launched_nccl->mut_mirrored_object());
#endif  // WITH_CUDA
  for (const auto& parameter : *parameters()) {
    if (!parameter) { continue; }
    DoEach(nullptr, CHECK_JUST(parameter->compute_local_dep_object())->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow
