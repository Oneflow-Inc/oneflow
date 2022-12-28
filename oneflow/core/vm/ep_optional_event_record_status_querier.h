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
#ifndef ONEFLOW_CORE_VM_EP_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_EP_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_

#include <atomic>
#include "oneflow/core/vm/ep_event.h"

namespace oneflow {

class DeviceCtx;

namespace vm {

class EpOptionalEventRecordStatusQuerier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpOptionalEventRecordStatusQuerier);
  ~EpOptionalEventRecordStatusQuerier();

  bool launched() const { return launched_; }

  bool done() const { return launched_ && (ep_event_ == nullptr || ep_event_->Query()); }

  void SetLaunched(ep::Stream* stream);

  void reset_ep_event(const std::shared_ptr<EpEvent>& ep_event) { ep_event_ = ep_event; }

  static const EpOptionalEventRecordStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const EpOptionalEventRecordStatusQuerier*>(mem_ptr);
  }
  static EpOptionalEventRecordStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<EpOptionalEventRecordStatusQuerier*>(mem_ptr);
  }
  static EpOptionalEventRecordStatusQuerier* PlacementNew(
      char* mem_ptr, const std::shared_ptr<EpEvent>& ep_event) {
    return new (mem_ptr) EpOptionalEventRecordStatusQuerier(ep_event);
  }

 private:
  explicit EpOptionalEventRecordStatusQuerier(const std::shared_ptr<EpEvent>& ep_event)
      : launched_(false), ep_event_(ep_event) {}

  std::atomic<bool> launched_;
  std::shared_ptr<EpEvent> ep_event_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_EP_OPTIONAL_EVENT_RECORD_STATUS_QUERIER_H_
