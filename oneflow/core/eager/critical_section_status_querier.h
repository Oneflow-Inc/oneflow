#ifndef ONEFLOW_CORE_EAGER_CRITICAL_SECTION_QUERIER_H_
#define ONEFLOW_CORE_EAGER_CRITICAL_SECTION_QUERIER_H_

#include <atomic>
#include <memory>
#include "oneflow/core/device/event_record.h"

namespace oneflow {
namespace vm {

class CriticalSectionStatusQuerier final {
 public:
  ~CriticalSectionStatusQuerier() = default;

  bool QueryDone() const { return launched_ && event_record_->QueryDone(); }

  void SetLaunched(const std::shared_ptr<EventRecord>& event_record) {
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
  explicit CriticalSectionStatusQuerier()
    : launched_(false) {}

  std::atomic<bool> launched_;
  std::shared_ptr<EventRecord> event_record_;
};

}
}

#endif  // ONEFLOW_CORE_EAGER_CRITICAL_SECTION_QUERIER_H_
