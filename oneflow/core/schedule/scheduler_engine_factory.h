#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULER_ENGINE_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_SCHEDULER_ENGINE_FACTORY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/node.h"
#include "oneflow/core/schedule/scheduler_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class SchedulerEngineFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SchedulerEngineFactory);
  SchedulerEngineFactory() = default;
  virtual ~SchedulerEngineFactory() = default;
  DEFINE_FACTORY_PURE_VIRTUAL_CLONE(SchedulerEngineFactory);

  virtual std::unique_ptr<SchedulerEngine> CreateSchedulerEngine(
      const Session& session,
      const std::function<uint32_t(uint64_t)>& get_regst_num) const = 0;
  virtual std::unique_ptr<SchedulerEngine> CreateSchedulerEngine(
      const Session& session, uint32_t regst_num_max = 3u) const = 0;
};

template<typename SchedulerEngineType>
class SchedulerEngineConcreteFactory : public SchedulerEngineFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SchedulerEngineConcreteFactory);
  SchedulerEngineConcreteFactory() = default;
  explicit SchedulerEngineConcreteFactory(ScheduleFactoryProvider*){};
  virtual ~SchedulerEngineConcreteFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(SchedulerEngineConcreteFactory,
                              SchedulerEngineFactory);

  std::unique_ptr<SchedulerEngine> CreateSchedulerEngine(
      const Session& session, uint32_t regst_num_max) const {
    return unique_ptr_new<SchedulerEngineType>(const_cast<Session*>(&session),
                                               regst_num_max);
  }

  std::unique_ptr<SchedulerEngine> CreateSchedulerEngine(
      const Session& session,
      const std::function<uint32_t(uint64_t)>& get_regst_num) const {
    return unique_ptr_new<SchedulerEngineType>(const_cast<Session*>(&session),
                                               get_regst_num);
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULER_ENGINE_FACTORY_H_
