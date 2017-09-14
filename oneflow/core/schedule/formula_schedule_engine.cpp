#include "oneflow/core/schedule/formula_schedule_engine.h"
namespace oneflow {
namespace schedule {

std::unique_ptr<Schedule> FormulaScheduleEngine::StaticSchedule() {
  auto schedule = of_make_unique<Schedule>(session());
  schedule->mut_max_interval() = EvaluateInitiationInterval();
  ForEachRegstDescDuration([&](SRegstDesc* regst_desc, float duration) {
    schedule->mut_regst_desc2duration()[regst_desc] = duration;
  });
  schedule->UpdateRegstCount();
  return schedule;
}

std::unique_ptr<Schedule> FormulaScheduleEngine::StaticSchedule(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  auto schedule = StaticSchedule();
  float initiation_interval = 0;
  sgraph().ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    uint32_t count = std::max(get_regst_num(regst_desc->id()), 1u);
    float ii = schedule->GetRegstDescDuration(regst_desc) / count;
    initiation_interval = std::max(initiation_interval, ii);
  });
  schedule->mut_max_interval() = initiation_interval;
  schedule->UpdateRegstCount();
  return schedule;
}

void FormulaScheduleEngine::ForEachRegstDescDuration(
    const std::function<void(SRegstDesc*, float)>& cb) {
  auto foreach_next = std::bind(&SGraph::ForeachNext, &sgraph(),
                                std::placeholders::_1, std::placeholders::_2);
  auto foreach_prev = std::bind(&SGraph::ForeachPrev, &sgraph(),
                                std::placeholders::_1, std::placeholders::_2);
  auto is_ascendant = [&](STask* asc, STask* node) {
    return sgraph().ascendant_arc_mgr().Find(node, asc) > 0u;
  };
  LongestPathVisitor<STask*> lpath(foreach_next, foreach_prev, is_ascendant);
  sgraph().ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    cb(regst_desc, GetRegstDescDuration(lpath, regst_desc));
  });
}

float FormulaScheduleEngine::GetRegstDescDuration(
    const LongestPathVisitor<STask*>& lpath, SRegstDesc* regst_desc) {
  auto get_node_weight = [&](STask* task) { return GetSTaskWeight(task); };
  float duration = 0;
  STask* owner = const_cast<STask*>(regst_desc->owner_task());
  sgraph().subscribed_regst_desc_mgr().Input(
      regst_desc, [&](STask* subscriber) {
        auto path_handler = [&](const std::list<STask*>& path) {
          if (path.back() == subscriber) {
            float d = 0;
            for (auto task : path) { d += get_node_weight(task); }
            duration = std::max(duration, d);
          }
        };
        lpath(owner, subscriber, get_node_weight, path_handler);
      });
  return duration;
}

float FormulaScheduleEngine::GetSTaskWeight(STask* task) {
  return static_cast<float>(1);
}

float FormulaScheduleEngine::EvaluateInitiationInterval() {
  return static_cast<float>(1);
}

}  // namespace schedule
}  // namespace oneflow
