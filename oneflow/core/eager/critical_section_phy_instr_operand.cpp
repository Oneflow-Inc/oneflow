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

namespace oneflow {
namespace vm {

CriticalSectionPhyInstrOperand::CriticalSectionPhyInstrOperand(
    const one::EagerBlobObjectListPtr& eager_blob_objects,
    const HashMap<std::string, int64_t>& op_name2index,
    const std::shared_ptr<NNGraphIf>& nn_graph)
      : eager_blob_objects_(eager_blob_objects), nn_graph_(nn_graph), notifier_(std::make_unique<Notifier>()), op_name2index_(op_name2index),consumer_ref_cnt_(std::make_shared<std::atomic<int64_t>>(eager_blob_objects->size())) {
}

void CriticalSectionPhyInstrOperand::ProducerNotifiesConsumer() const {
  CHECK(notifier_->Notify() == kNotifierStatusSuccess);
}

void CriticalSectionPhyInstrOperand::ConsumerWaitsProducer() const {
  notifier_->WaitAndClearNotifiedCnt();
}

void CriticalSectionPhyInstrOperand::ConsumerFetchBlobAndDecreaseRefCnt(const std::string& op_name, const std::function<void(Blob*)>& Callback) const {
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

void CriticalSectionPhyInstrOperand::ForEachMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach) const {
  for (const auto& eager_blob_object : *eager_blob_objects_) {
    DoEach(nullptr, CHECK_JUST(eager_blob_object->compute_local_dep_object())->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow
