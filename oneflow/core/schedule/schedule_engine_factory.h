#ifndef ONEFLOW_CORE_SCHEDULE_SCHEDULER_ENGINE_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_SCHEDULER_ENGINE_FACTORY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/schedule_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class ScheduleEngineFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleEngineFactory);
  ScheduleEngineFactory() = default;
  virtual ~ScheduleEngineFactory() = default;
  DEFINE_FACTORY_PURE_VIRTUAL_CLONE(ScheduleEngineFactory);

  virtual std::unique_ptr<ScheduleEngine> CreateScheduleEngine(
      const Session& session) const = 0;
};

template<typename ScheduleEngineType>
class ScheduleEngineConcreteFactory : public ScheduleEngineFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleEngineConcreteFactory);
  ScheduleEngineConcreteFactory() = default;
  explicit ScheduleEngineConcreteFactory(ScheduleFactoryProvider*){};
  virtual ~ScheduleEngineConcreteFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(ScheduleEngineConcreteFactory,
                              ScheduleEngineFactory);

  std::unique_ptr<ScheduleEngine> CreateScheduleEngine(
      const Session& session) const {
    return of_make_unique<ScheduleEngineType>(const_cast<Session*>(&session));
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SCHEDULER_ENGINE_FACTORY_H_
