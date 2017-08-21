#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACADE_H
#define ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACADE_H

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"

namespace oneflow {
namespace schedule {

class ScheduleFacade final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleFacade);
  ScheduleFacade(const std::string& name)
      : sfp_(ScheduleFactoryConfigure::Provider(name)) {}

  static ScheduleFacade* Default() {
    static ScheduleFacade* facade = new ScheduleFacade(DefaultName());
    return facade;
  }

  void Allocate(Plan* plan) {
    auto allocator = sfp_->allocator_factory()->CreateAllocator();
    allocator->Allocate(plan);
  }

 private:
  static std::string DefaultName() { return "default"; }
  ScheduleFactoryProvider* sfp_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULE_FACADE_H
