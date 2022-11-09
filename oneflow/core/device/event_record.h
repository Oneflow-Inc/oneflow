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
#ifndef ONEFLOW_CORE_DEVICE_EVENT_RECORD_H_
#define ONEFLOW_CORE_DEVICE_EVENT_RECORD_H_

#include <atomic>
#include <memory>
#include "oneflow/core/common/util.h"

namespace oneflow {

class EventRecord {
 public:
  EventRecord(const EventRecord&) = delete;
  EventRecord(EventRecord&&) = delete;
  EventRecord& operator=(const EventRecord&) = delete;
  EventRecord& operator=(EventRecord&&) = delete;
  virtual ~EventRecord() = default;

  virtual bool QueryDone() const = 0;

  EventRecord() = default;
};

class NaiveEventRecord final : public EventRecord {
 public:
  NaiveEventRecord(const NaiveEventRecord&) = delete;
  NaiveEventRecord(NaiveEventRecord&&) = delete;
  NaiveEventRecord& operator=(const NaiveEventRecord&) = delete;
  NaiveEventRecord& operator=(NaiveEventRecord&&) = delete;

  NaiveEventRecord() = default;
  ~NaiveEventRecord() override = default;

  bool QueryDone() const override { return true; }
};

class SharedEventRecord final : public EventRecord {
 public:
  SharedEventRecord(const SharedEventRecord&) = delete;
  SharedEventRecord(SharedEventRecord&&) = delete;
  SharedEventRecord& operator=(const SharedEventRecord&) = delete;
  SharedEventRecord& operator=(SharedEventRecord&&) = delete;

  SharedEventRecord() : EventRecord(), inited_(false) {}
  ~SharedEventRecord() override = default;

  bool QueryDone() const override { return inited_ && event_record_->QueryDone(); }

  void Init(const std::shared_ptr<EventRecord>& event_record) {
    // No lock needed. This function will be called only one time.
    // In most cases, errors will be successfully detected by CHECK
    // even though run in different threads.
    CHECK(!inited_);
    event_record_ = event_record;
    inited_ = true;
  }
  void TryInit(const std::shared_ptr<EventRecord>& event_record) {
    if (!inited_) { Init(event_record); }
  }

 private:
  std::atomic<bool> inited_;
  std::shared_ptr<EventRecord> event_record_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_EVENT_RECORD_H_
