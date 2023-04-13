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
#ifndef ONEFLOW_CORE_VM_CRITICAL_SECTION_QUERIER_H_
#define ONEFLOW_CORE_VM_CRITICAL_SECTION_QUERIER_H_

#include <atomic>
#include <memory>
#include "oneflow/core/device/event_record.h"

namespace oneflow {
namespace vm {

class CriticalSectionStatusQuerier final {
 public:
  ~CriticalSectionStatusQuerier() = default;

  bool QueryLaunched() const { return launched_; }
  bool QueryDone() const { return launched_ && event_record_->QueryDone(); }

  void SetLaunched(const std::shared_ptr<EventRecord>& event_record) {
    // No lock needed. This function will be called only one time.
    // In most cases, errors will be successfully detected by CHECK
    // even though run in different threads.
    CHECK(!launched_);
    event_record_ = event_record;
    launched_ = true;
  }

  static const CriticalSectionStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const CriticalSectionStatusQuerier*>(mem_ptr);
  }
  static CriticalSectionStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<CriticalSectionStatusQuerier*>(mem_ptr);
  }
  static CriticalSectionStatusQuerier* PlacementNew(char* mem_ptr) {
    return new (mem_ptr) CriticalSectionStatusQuerier();
  }

 private:
  explicit CriticalSectionStatusQuerier() : launched_(false) {}

  std::atomic<bool> launched_;
  std::shared_ptr<EventRecord> event_record_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CRITICAL_SECTION_QUERIER_H_
