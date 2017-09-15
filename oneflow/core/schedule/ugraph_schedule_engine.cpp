#include "oneflow/core/schedule/ugraph_schedule_engine.h"
namespace oneflow {
namespace schedule {

float UGraphScheduleEngine::GetSTaskWeight(STask* task) const {
  return task2weight_.at(task);
}

float UGraphScheduleEngine::EvaluateInitiationInterval() const {
  return initiation_interval_;
}

void UGraphScheduleEngine::Init() {
  InitTaskWeight();
  InitII();
}

void UGraphScheduleEngine::InitTaskWeight() {
  session().ugraph().node_mgr<TaskUtilization>().ForEach(
      [&](TaskUtilization* tu) {
        STask* task = session().sgraph().node_mgr().Find(tu->task_id());
        CHECK(task);
        task2weight_[task] = tu->GetDuration(session().ugraph());
      });
}

void UGraphScheduleEngine::InitII() {
  initiation_interval_ = 0;
  session().ugraph().node_mgr<StreamUtilization>().ForEach(
      [&](StreamUtilization* su) {
        initiation_interval_ =
            std::max(initiation_interval_,
                     su->GetInitiationInterval(session().ugraph()));
      });
}

}  // namespace schedule
}  // namespace oneflow
