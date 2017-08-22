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
  void InitRegstDesc(const Plan& plan);
  void InitTask(const Plan& plan);
  void InitLoss(const Plan& plan);
  void GenerateOpName2IsLoss(
      const Plan& plan,
      std::unordered_map<std::string, bool>* op_name2is_loss) const;
  void GenerateTaskId2IsLoss(
      const Plan& plan,
      std::unordered_map<int64_t, bool>* task_id2is_loss) const;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_PLAN_SGRAPH_H_
