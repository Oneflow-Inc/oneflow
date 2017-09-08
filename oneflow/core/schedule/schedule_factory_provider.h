#ifndef ONEFLOW_CORE_SCHEDULE_FACTORY_PROVIDER_H_
#define ONEFLOW_CORE_SCHEDULE_FACTORY_PROVIDER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/allocator_factory.h"
#include "oneflow/core/schedule/schedule_engine_factory.h"
#include "oneflow/core/schedule/session_factory.h"
#include "oneflow/core/schedule/sgraph_factory.h"
#include "oneflow/core/schedule/utilization_analyzer_factory.h"
#include "oneflow/core/schedule/validator_factory.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleFactoryProvider);
  explicit ScheduleFactoryProvider(const std::string name) : name_(name) {}

  ScheduleFactoryProvider* Merge(const ScheduleFactoryProvider* sfp) {
    CLONE_FACTORY(sfp, sgraph_factory());
    CLONE_FACTORY(sfp, session_factory());
    CLONE_FACTORY(sfp, schedule_engine_factory());
    CLONE_FACTORY(sfp, validator_factory());
    CLONE_FACTORY(sfp, allocator_factory());
    CLONE_FACTORY(sfp, utilization_analyzer_factory());
    return this;
  }

  ScheduleFactoryProvider* Set(std::unique_ptr<SGraphFactory>&& factory) {
    sgraph_factory_ = std::move(factory);
    return this;
  }
  ScheduleFactoryProvider* Set(std::unique_ptr<SessionFactory>&& factory) {
    session_factory_ = std::move(factory);
    return this;
  }
  ScheduleFactoryProvider* Set(
      std::unique_ptr<ScheduleEngineFactory>&& factory) {
    schedule_engine_factory_ = std::move(factory);
    return this;
  }
  ScheduleFactoryProvider* Set(std::unique_ptr<ValidatorFactory>&& factory) {
    validator_factory_ = factory->Clone(this);
    return this;
  }
  ScheduleFactoryProvider* Set(std::unique_ptr<AllocatorFactory>&& factory) {
    allocator_factory_ = factory->Clone(this);
    return this;
  }
  ScheduleFactoryProvider* Set(
      std::unique_ptr<UtilizationAnalyzerFactory>&& factory) {
    utilization_analyzer_factory_ = factory->Clone(this);
    return this;
  }

  //	getter
  inline const SGraphFactory* sgraph_factory() const {
    CHECK(sgraph_factory_.get());
    return sgraph_factory_.get();
  }
  inline const SessionFactory* session_factory() const {
    CHECK(session_factory_.get());
    return session_factory_.get();
  }
  inline const ScheduleEngineFactory* schedule_engine_factory() const {
    CHECK(schedule_engine_factory_.get());
    return schedule_engine_factory_.get();
  }
  inline const ValidatorFactory* validator_factory() const {
    CHECK(validator_factory_.get());
    return validator_factory_.get();
  }
  inline const AllocatorFactory* allocator_factory() const {
    CHECK(allocator_factory_.get());
    return allocator_factory_.get();
  }
  inline const UtilizationAnalyzerFactory* utilization_analyzer_factory()
      const {
    CHECK(utilization_analyzer_factory_.get());
    return utilization_analyzer_factory_.get();
  }

 private:
  //	setter
  inline std::unique_ptr<SGraphFactory>& mut_sgraph_factory() {
    return sgraph_factory_;
  }
  inline std::unique_ptr<SessionFactory>& mut_session_factory() {
    return session_factory_;
  }
  inline std::unique_ptr<ScheduleEngineFactory>& mut_schedule_engine_factory() {
    return schedule_engine_factory_;
  }
  inline std::unique_ptr<ValidatorFactory>& mut_validator_factory() {
    return validator_factory_;
  }
  inline std::unique_ptr<AllocatorFactory>& mut_allocator_factory() {
    return allocator_factory_;
  }
  inline std::unique_ptr<UtilizationAnalyzerFactory>&
  mut_utilization_analyzer_factory() {
    return utilization_analyzer_factory_;
  }
  std::string name_;
  std::unique_ptr<SGraphFactory> sgraph_factory_;
  std::unique_ptr<SessionFactory> session_factory_;
  std::unique_ptr<ScheduleEngineFactory> schedule_engine_factory_;
  std::unique_ptr<ValidatorFactory> validator_factory_;
  std::unique_ptr<AllocatorFactory> allocator_factory_;
  std::unique_ptr<UtilizationAnalyzerFactory> utilization_analyzer_factory_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_FACTORY_PROVIDER_H_
