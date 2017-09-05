#include "oneflow/core/schedule/formula_schedule_engine.h"
namespace oneflow {
namespace schedule {

std::unique_ptr<Schedule> FormulaScheduleEngine::StaticSchedule() {
  auto schedule = of_make_unique<Schedule>(session());
  return schedule;
}

std::unique_ptr<Schedule> FormulaScheduleEngine::StaticSchedule(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  auto schedule = of_make_unique<Schedule>(session());
  return schedule;
}

}  // namespace schedule
}  // namespace oneflow
