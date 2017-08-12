#ifndef ONEFLOW_CORE_SCHEDULE_ALLOCATOR_H_
#define ONEFLOW_CORE_SCHEDULE_ALLOCATOR_H_

#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/session.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class Allocator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Allocator);
  explicit Allocator(ScheduleFactoryProvider* schedule_factory_provider)
      : schedule_factory_provider_(schedule_factory_provider) {}
  Allocator() = default;

  void Allocate(Plan* plan) {}

  //	getter
  inline const ScheduleFactoryProvider* schedule_factory_provider() const {
    return schedule_factory_provider_;
  }

 private:
  ScheduleFactoryProvider* schedule_factory_provider_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_ALLOCATOR_H_
