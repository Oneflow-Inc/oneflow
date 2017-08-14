#include "oneflow/core/schedule/validator.h"
#include "oneflow/core/schedule/schedule_engine_factory.h"
#include "oneflow/core/schedule/schedule_factory_provider.h"

namespace oneflow {
namespace schedule {

bool Validator::ValidateMemory(const Schedule& schedule) {
  std::unordered_map<const SDevice*, uint64_t> device2total_memory_size;
  auto graph = schedule.session()->graph();
  graph->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    auto device = regst_desc->owner_task()->device();
    auto regst_count = GetOrDefault(schedule.regst_desc2count(), regst_desc, 0);
    auto memory_size = regst_count * regst_desc->regst_memory_size();
    device2total_memory_size[device] += memory_size;
  });

  for (const auto& p : device2total_memory_size) {
    if (p.second > p.first->memory_limit()) return false;
  }
  return true;
}

bool Validator::ValidateAllocation(const Schedule& schedule) {
  auto sess = schedule.session();
  auto engine_factory = schedule_factory_provider()->schedule_engine_factory();
  auto schedule_engine = engine_factory->CreateScheduleEngine(*sess);
  int target = 0;
  int failed = 0;
  for (int i = 0; i < schedule.regst_desc2count().size(); i++) {
    std::unordered_map<uint64_t, uint32_t> limited;
    int count = 0;
    bool declined = false;
    for (const auto& p : schedule.regst_desc2count()) {
      limited[p.first->id()] = p.second;
      if (count == target && limited[p.first->id()] > 1) {
        limited[p.first->id()] -= 1;
        declined = true;
      }
      count++;
    }
    auto get_regst_num = [&](uint64_t id) { return limited[id]; };
    target++;
    if (declined) {
      std::cout << "---------------" << std::endl;
      auto limited_schedule = schedule_engine->StaticSchedule(get_regst_num);
      if (limited_schedule->max_interval() <= schedule.max_interval()) {
        failed++;
      }
    }
  }
  return failed == 0;
}

}  // namespace schedule
}  // namespace oneflow
