#ifndef ONEFLOW_CORE_SCHEDULE_VALIDATOR_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_VALIDATOR_FACTORY_H_

#include "oneflow/core/schedule/validator.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class ValidatorFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ValidatorFactory);
  explicit ValidatorFactory(
      const ScheduleFactoryProvider& schedule_factory_provider)
      : schedule_factory_provider_(&schedule_factory_provider) {}
  ValidatorFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(ValidatorFactory, ValidatorFactory);

  inline const ScheduleFactoryProvider& schedule_factory_provider() const {
    return *schedule_factory_provider_;
  }

  virtual std::unique_ptr<Validator> CreateValidator() const {
    return of_make_unique<Validator>(*schedule_factory_provider_);
  }

 private:
  const ScheduleFactoryProvider* schedule_factory_provider_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_VALIDATOR_FACTORY_H_
