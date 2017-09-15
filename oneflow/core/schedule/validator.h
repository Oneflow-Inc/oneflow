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
  bool ValidateSGraph(const SGraph& sgraph) const {
    ValidateSGraphNode(sgraph);
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
  void ValidateSGraphNode(const SGraph& sgraph) const;
  virtual bool ValidateGraphArc(const SGraph& sgraph) const;
  ScheduleFactoryProvider* schedule_factory_provider_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_VALIDATOR_H_
