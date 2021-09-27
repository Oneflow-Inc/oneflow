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
  ~EventRecord() = default;

  virtual bool QueryDone() const = 0;

  EventRecord() = default;
};

class NaiveEventRecord final : public EventRecord {
 public:
   NaiveEventRecord(const NaiveEventRecord&) = delete;
   NaiveEventRecord(NaiveEventRecord&&) = delete;
   NaiveEventRecord& operator=(const NaiveEventRecord&) = delete;
   NaiveEventRecord& operator=(NaiveEventRecord&&) = delete;

  using EventRecord::EventRecord;
  ~NaiveEventRecord() = default;

  bool QueryDone() const { return true; } 
};

class SharedEventRecord final : public EventRecord {
 public:
  SharedEventRecord(const SharedEventRecord&) = delete;
  SharedEventRecord(SharedEventRecord&&) = delete;
  SharedEventRecord& operator=(const SharedEventRecord&) = delete;
  SharedEventRecord& operator=(SharedEventRecord&&) = delete;

  SharedEventRecord(): EventRecord(), inited_(false) {}
  ~SharedEventRecord() = default;

  bool QueryDone() const override { return inited_ && event_record_->QueryDone(); }

  void Init(const std::shared_ptr<EventRecord>& event_record) {
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

}

#endif  // ONEFLOW_CORE_DEVICE_EVENT_RECORD_H_ 
