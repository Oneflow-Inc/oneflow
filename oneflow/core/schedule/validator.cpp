#include "oneflow/core/schedule/validator.h"
#include "oneflow/core/schedule/scc_visitor.h"
#include "oneflow/core/schedule/schedule_engine_factory.h"
#include "oneflow/core/schedule/schedule_factory_provider.h"

namespace oneflow {
namespace schedule {

bool Validator::ValidateGraphArc(
    const SGraph& sgraph, const std::function<void(const Arc<STask>&)>& cb) {
  typedef std::function<void(STask * task)> TaskVisitor;
  auto foreach_next = [&](STask* task, const TaskVisitor& cb) {
    sgraph.arc_mgr().Output(task, cb);
  };
  SccVisitor<STask*> scc(foreach_next);
  auto print_component = [&](const std::list<STask*> l) {
    for (STask* task : l) {
      if (task != l.front()) { std::cout << ","; }
      std::cout << task->id();
    }
    std::cout << std::endl;
  };
  uint32_t scc_cnt = scc(sgraph.source(), print_component);
  CHECK(scc_cnt == 0u);

  sgraph.ForeachArc([&](TaskArc* arc) {
    CHECK(arc->dst_node() != sgraph.source());
    CHECK(arc->src_node() != sgraph.sink());
  });
  return true;
}

bool Validator::ValidateMemory(const Schedule& schedule) {
  std::unordered_map<const SDevice*, uint64_t> device2total_memory_size;
  schedule.sgraph().ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    const SDevice* device = regst_desc->owner_task()->device();
    uint32_t regst_count =
        GetOrDefault(schedule.regst_desc2count(), regst_desc, 0u);
    uint32_t memory_size = regst_count * regst_desc->regst_memory_size();
    device2total_memory_size[device] += memory_size;
  });

  for (const auto& p : device2total_memory_size) {
    if (p.second > p.first->memory_limit()) return false;
  }
  return true;
}

bool Validator::ValidateAllocation(const Schedule& schedule) {
  auto engine_factory = schedule_factory_provider()->schedule_engine_factory();
  auto schedule_engine =
      engine_factory->CreateScheduleEngine(schedule.session());
  uint32_t target = 0;
  uint32_t failed = 0;

  std::cout << "origin ii = " << schedule.max_interval() << std::endl;
  std::cout << "-------[ more ]-------" << std::endl;
  std::unordered_map<uint64_t, uint32_t> more_regst;
  for (const auto& p : schedule.regst_desc2count()) {
    more_regst[p.first->id()] = p.second + 1;
  }
  auto get_regst_num = [&](uint64_t id) { return more_regst[id]; };
  std::unique_ptr<Schedule> more_memory_schedule =
      schedule_engine->StaticSchedule(get_regst_num);
  std::cout << "ii = " << more_memory_schedule->max_interval() << std::endl;
  if (more_memory_schedule->max_interval() > schedule.max_interval() * 1.1) {
    failed++;
    more_memory_schedule->PrintRegstNum();
    more_memory_schedule->PrintSchedule();
  }

  for (uint32_t i = 0; i < schedule.regst_desc2count().size(); i++) {
    std::unordered_map<uint64_t, uint32_t> limited;
    uint32_t count = 0;
    bool declined = false;
    for (const auto& p : schedule.regst_desc2count()) {
      limited[p.first->id()] = p.second;
      if (count == target && limited[p.first->id()] > 1) {
        limited[p.first->id()] -= 1;
        declined = true;
        std::cout << "-------[ less ]--------" << std::endl;
        std::cout << p.first->id() << ": " << limited[p.first->id()]
                  << std::endl;
      }
      count++;
    }
    auto get_regst_num = [&](uint64_t id) { return limited[id]; };
    target++;
    if (declined) {
      std::unique_ptr<Schedule> limited_schedule =
          schedule_engine->StaticSchedule(get_regst_num);
      std::cout << "ii = " << limited_schedule->max_interval() << std::endl;
      if (limited_schedule->max_interval() <= schedule.max_interval()) {
        failed++;
        limited_schedule->PrintRegstNum();
        limited_schedule->PrintSchedule();
      }
    }
  }
  return failed == 0;
}

}  // namespace schedule
}  // namespace oneflow
