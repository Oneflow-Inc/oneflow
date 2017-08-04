#ifndef ONEFLOW_CORE_SCHEDULE_INTERFACE_SCHEDULER_H_
#define ONEFLOW_CORE_SCHEDULE_INTERFACE_SCHEDULER_H_

#include "oneflow/core/schedule/facotry/factory.h"
#include "oneflow/core/schedule/interface/policy.h"
#include "oneflow/core/schedule/interface/policy_hub.h"

namespace oneflow {
namespace schedule {

class Scheduler {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Scheduler);
  virtual ~Scheduler() = default;

  static void Allocate(Plan* plan) {
    PolicyHub* ph = DefaultPH();
    CHECK(ph);
    ph->Allocate(plan);
  }

 private:
  static PolicyHub* DefaultPH() { return PH("default"); }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_INTERFACE_SCHEDULER_H_
