#include "oneflow/core/schedule/allocator.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
namespace oneflow {
namespace schedule {
std::unique_ptr<Schedule> Allocator::MemoryLimitedStaticSchedule(
    const Session& session) {
  auto engine_factory = schedule_factory_provider()->schedule_engine_factory();
  auto validator_factory = schedule_factory_provider()->validator_factory();

  auto schedule_engine = engine_factory->CreateScheduleEngine(session);
  auto validator = validator_factory->CreateValidator();

  auto schedule = schedule_engine->StaticSchedule();
  uint32_t max_regst_count = schedule->max_regst_count();
  auto get_regst_num = [&](uint64_t) { return max_regst_count; };
  while (max_regst_count > 0 && !validator->ValidateMemory(*schedule)) {
    schedule = schedule_engine->StaticSchedule(get_regst_num);
    --max_regst_count;
  }
  return std::move(schedule);
}

}  // namespace schedule
}  // namespace oneflow
