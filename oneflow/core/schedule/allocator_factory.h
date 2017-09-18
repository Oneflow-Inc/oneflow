#ifndef ONEFLOW_CORE_SCHEDULE_ALLOCATOR_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_ALLOCATOR_FACTORY_H_

#include "oneflow/core/schedule/allocator.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class AllocatorFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AllocatorFactory);
  explicit AllocatorFactory(
      const ScheduleFactoryProvider& schedule_factory_provider)
      : schedule_factory_provider_(&schedule_factory_provider) {}
  AllocatorFactory() = default;
  virtual ~AllocatorFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(AllocatorFactory, AllocatorFactory);

  virtual std::unique_ptr<Allocator> CreateAllocator() const {
    return of_make_unique<Allocator>(*schedule_factory_provider_);
  }

  //	getter
  inline const ScheduleFactoryProvider& sfp() const {
    return *schedule_factory_provider_;
  }
  inline const ScheduleFactoryProvider& schedule_factory_provider() const {
    return *schedule_factory_provider_;
  }

 private:
  const ScheduleFactoryProvider* schedule_factory_provider_;
};

template<typename AllocatorType>
class AllocatorConcreteFactory : public AllocatorFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AllocatorConcreteFactory);
  explicit AllocatorConcreteFactory(const ScheduleFactoryProvider& sfp)
      : AllocatorFactory(sfp) {}
  AllocatorConcreteFactory() = default;
  virtual ~AllocatorConcreteFactory() = default;
  virtual std::unique_ptr<Allocator> CreateAllocator() const {
    return of_make_unique<AllocatorType>(schedule_factory_provider());
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_ALLOCATOR_FACTORY_H_
