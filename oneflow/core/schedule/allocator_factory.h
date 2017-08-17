#ifndef ONEFLOW_CORE_SCHEDULE_ALLOCATOR_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_ALLOCATOR_FACTORY_H_

#include "oneflow/core/schedule/allocator.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class AllocatorFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AllocatorFactory);
  explicit AllocatorFactory(ScheduleFactoryProvider* schedule_factory_provider)
      : schedule_factory_provider_(schedule_factory_provider) {}
  AllocatorFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(AllocatorFactory, AllocatorFactory);

  virtual std::unique_ptr<Allocator> CreateAllocator() const {
    return unique_ptr_new<Allocator>(schedule_factory_provider_);
  }

  //	getter
  inline const ScheduleFactoryProvider* schedule_factory_provider() const {
    return schedule_factory_provider_;
  }

 private:
  ScheduleFactoryProvider* schedule_factory_provider_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_ALLOCATOR_FACTORY_H_
