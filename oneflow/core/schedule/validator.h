#ifndef ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_
#define ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class Validator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Validator);
  explicit Validator(ScheduleFactoryProvider* schedule_factory_provider)
      : schedule_factory_provider_(schedule_factory_provider) {}
  Validator() = default;

  //	graph
  virtual bool ValidateGraphArc(
      const SGraph& sgraph,
      const std::function<void(const Arc<STask>&)>& cb) const;
  bool ValidateGraphArc(const SGraph& sgraph) const {
    return ValidateGraphArc(sgraph, [](const Arc<STask>&) {});
  }
  bool ValidateSGraph(const SGraph& sgraph) const {
    return ValidateGraphArc(sgraph);
  }
  bool ValidateUtilizationGraph(const UtilizationGraph& ugraph) const;

  //	allocation
  virtual bool ValidateAllocation(const Schedule& schedule) const;

  //	memory
  virtual bool ValidateMemory(const Schedule& schedule) const;

  //	getter
  inline ScheduleFactoryProvider& sfp() const {
    return *schedule_factory_provider_;
  }
  inline ScheduleFactoryProvider& schedule_factory_provider() const {
    return *schedule_factory_provider_;
  }

 private:
  ScheduleFactoryProvider* schedule_factory_provider_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_
