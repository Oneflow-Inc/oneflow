#ifndef ONEFLOW_CORE_SCHEDULE_PLAN_SGRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_PLAN_SGRAPH_H_

#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

class PlanSGraph final : public SGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PlanSGraph);
  virtual ~PlanSGraph() = default;
  explicit PlanSGraph(const Plan& plan) : SGraph(plan) {
    InitSourceAndSink();
    InitFromPlan(plan);
    Update();
  }

 private:
  void InitFromPlan(const Plan& plan);
  void InitDevice();
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_PLAN_SGRAPH_H_
