#ifndef ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_
#define ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_

#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class Validator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Validator);
  explicit Validator(ScheduleFactoryProvider* schedule_factory_provider)
      : schedule_factory_provider_(schedule_factory_provider) {}
  Validator() = default;

  virtual bool ValidateAllocation(const Schedule& schedule);

  //	getter
  inline ScheduleFactoryProvider* schedule_factory_provider() const {
    return schedule_factory_provider_;
  }

 private:
  ScheduleFactoryProvider* schedule_factory_provider_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_
