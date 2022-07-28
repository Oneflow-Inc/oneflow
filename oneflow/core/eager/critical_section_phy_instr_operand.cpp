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
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/ep_based_event_record.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {
namespace vm {

void CriticalSectionBeginPhyInstrOperand::ForEachDependence(
    const std::function<void(vm::Dependence* compute)>& DoEach) const {
  for (const auto& eager_blob_object : *eager_blob_objects_) {
    DoEach(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
  }
}

void CriticalSectionEndPhyInstrOperand::ForEachDependence(
    const std::function<void(vm::Dependence* compute)>& DoEach) const {
  DoEach(CHECK_JUST(eager_blob_object_->compute_local_dep_object()));
}

void CriticalSectionBeginPhyInstrOperand::ForEachMutDependence(
    const std::function<void(vm::Dependence* compute)>& DoEach) const {
  DoEach(vm_stream_->schedule_local_dep_object().get());
}

void CriticalSectionBeginPhyInstrOperand::FinishInvalidInterfaceEventRecords() {
  for (const auto& op_name : interfaces_op_names()) {
    size_t index = CHECK_JUST(MapAt(op_name2interface_index_, op_name));
    if (!interfaces_valid().at(index)) {
      const auto& iter = op_name2end_event_record_->find(op_name);
      CHECK(iter != op_name2end_event_record_->end());
      iter->second->Init(std::make_shared<NaiveEventRecord>());
    }
  }
}

void CriticalSectionBeginPhyInstrOperand::Finish() {
  for (const auto& pair : *op_name2end_event_record_) {
    pair.second->TryInit(std::make_shared<NaiveEventRecord>());
  }
}

void InputCriticalSectionBeginPhyInstrOperand::AccessBlobByOpName(ep::Stream* stream, Blob* blob,
                                                                  const std::string& op_name) {
  int64_t i = CHECK_JUST(MapAt(op_name2interface_index_, op_name));
  CHECK(interfaces_valid().at(i));
  const auto& eager_blob_object = eager_blob_objects_->at(i);
  {
    size_t header_size = blob->blob_desc().ByteSizeOfBlobHeader();
    CHECK_EQ(header_size, eager_blob_object->shape().NumAxes() * sizeof(int64_t));
    CHECK_EQ(blob->static_shape(), eager_blob_object->shape());
  }
  const auto& end_event_record = op_name2end_event_record_->at(op_name);
  if (eager_blob_object->dptr() == nullptr) {
    end_event_record->Init(std::make_shared<NaiveEventRecord>());
  } else {
    {
      const size_t body_bytes = blob->ByteSizeOfBlobBody();
      CHECK_EQ(eager_blob_object->ByteSizeOfBlobBody(), body_bytes);
      AutoMemcpy(stream, blob->mut_dptr(), eager_blob_object->dptr(),
                 body_bytes, blob->mem_case(), eager_blob_object->mem_case());
    }
    end_event_record->Init(EpBasedEventRecord::MakeEventRecord(stream));
  }
}

void OutputCriticalSectionBeginPhyInstrOperand::AccessBlobByOpName(ep::Stream* stream, Blob* blob,
                                                                   const std::string& op_name) {
  int64_t i = CHECK_JUST(MapAt(op_name2interface_index_, op_name));
  CHECK(interfaces_valid().at(i));
  auto& eager_blob_object = eager_blob_objects_->at(i);
  CHECK_EQ(blob->static_shape(), eager_blob_object->shape());
  const auto& end_event_record = op_name2end_event_record_->at(op_name);
  if (eager_blob_object->dptr() == nullptr) {
    end_event_record->Init(std::make_shared<NaiveEventRecord>());
  } else {
    {
      const size_t body_bytes = blob->ByteSizeOfBlobBody();
      CHECK_EQ(eager_blob_object->ByteSizeOfBlobBody(), body_bytes);
      AutoMemcpy(stream, eager_blob_object->mut_dptr(), blob->dptr(),
                 body_bytes, eager_blob_object->mem_case(), blob->mem_case());
    }
    end_event_record->Init(EpBasedEventRecord::MakeEventRecord(stream));
  }
}

void CriticalSectionEndPhyInstrOperand::ForEachMutDependence(
    const std::function<void(vm::Dependence* compute)>& DoEach) const {
  DoEach(vm_stream_->schedule_local_dep_object().get());
}

}  // namespace vm
}  // namespace oneflow
