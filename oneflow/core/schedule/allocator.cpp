#include "oneflow/core/schedule/allocator.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/utilization_graph.h"
namespace oneflow {
namespace schedule {

void Allocator::Allocate(Plan* plan,
                         const std::string& dev_info_proto_log_file) const {
  const auto& sgraph_factory = schedule_factory_provider()->sgraph_factory();
  const auto& analyzer_factory =
      schedule_factory_provider()->utilization_analyzer_factory();
  const auto& session_factory = schedule_factory_provider()->session_factory();
  const auto& validator_factory =
      schedule_factory_provider()->validator_factory();
  std::unique_ptr<Validator> validator = validator_factory.CreateValidator();

  std::unique_ptr<SGraph> sgraph = sgraph_factory.CreateSGraph(*plan);
  CHECK(validator->ValidateGraph(*sgraph));

  std::unique_ptr<UtilizationAnalyzer> analyzer =
      analyzer_factory.CreateUtilizationAnalyzer(*sgraph);

  std::unique_ptr<UtilizationGraph> ugraph =
      analyzer->CreateUtilizationGraph(dev_info_proto_log_file);
  std::unique_ptr<Session> session =
      session_factory.CreateSession(*sgraph, *ugraph);
  std::unique_ptr<Schedule> schedule = MemoryLimitedStaticSchedule(*session);
  SetRegstNum(*schedule, plan);
}

void Allocator::SetRegstNum(const Schedule& schedule, Plan* plan) const {
  if (!plan) return;
  auto get_regst_num = [&](int64_t id) {
    SRegstDesc* regst_desc = schedule.sgraph().regst_desc_mgr().Find(id);
    uint32_t count = GetOrDefault(schedule.regst_desc2count(), regst_desc, 2u);
    count = std::max(count, regst_desc->min_regst_count());
    return count;
  };
  for (auto& task_proto : plan->task()) {
    for (auto& pair : task_proto.produced_regst_desc()) {
      auto regst_desc = const_cast<RegstDescProto*>(&pair.second);
      uint32_t regst_num = get_regst_num(regst_desc->regst_desc_id());
      regst_desc->set_register_num(regst_num);
    }
  }
}

std::unique_ptr<Schedule> Allocator::MemoryLimitedStaticSchedule(
    const Session& session) const {
  const auto& engine_factory =
      schedule_factory_provider()->schedule_engine_factory();
  const auto& validator_factory =
      schedule_factory_provider()->validator_factory();

  std::unique_ptr<ScheduleEngine> schedule_engine =
      engine_factory.CreateScheduleEngine(session);
  std::unique_ptr<Validator> validator = validator_factory.CreateValidator();

  std::unique_ptr<Schedule> schedule = schedule_engine->StaticSchedule();
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
