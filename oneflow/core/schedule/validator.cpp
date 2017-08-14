#include "oneflow/core/schedule/validator.h"
#include "oneflow/core/schedule/policy_hub.h"
#include "oneflow/core/schedule/schedule_factory_provider.h"
#include "oneflow/core/schedule/scheduler_engine_factory.h"
#include "oneflow/core/schedule/simulator_session.h"
namespace oneflow {
namespace schedule {

bool Validator::ValidateAllocation(const Schedule& schedule) {
  auto sess = schedule.session();
  auto engine_factory = schedule_factory_provider()->scheduler_engine_factory();
  auto scheduler_engine = engine_factory->CreateSchedulerEngine(*sess);
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
      auto limited_schedule = scheduler_engine->StaticSchedule(get_regst_num);
      if (limited_schedule->max_interval() <= schedule.max_interval()) {
        failed++;
      }
    }
  }
  return failed == 0;
}

bool AllocationValidatorSimplePolicy::ValidateAllocation(
    const Session& session, const ScheduleResult& result) {
  auto sess_ptr = const_cast<Session*>(&session);
  auto sess = dynamic_cast<SimulatorSession*>(sess_ptr);
  int target = 0;
  int failed = 0;
  for (int i = 0; i < result.regst_desc2count().size(); i++) {
    std::unordered_map<uint64_t, uint32_t> limited;
    int count = 0;
    bool declined = false;
    for (const auto& p : result.regst_desc2count()) {
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
      LimitedMode<NegativeStrategy> m3(sess, get_regst_num);
      m3.Run();
      auto log = sess->GetLoggerThenReset();

      LimitedMode<PositiveStrategy> m4(sess, get_regst_num);
      m4.Run();

      sess->logger()->MergeTimeGapToLossInPlace(&*log);
      sess->logger()->UpdateDuration(sess, &m4);
      auto logger = sess->GetLoggerThenReset();
      ph()->AllocateFromSchedule(session, logger.get());
      if (logger->max_interval() <= result.max_interval()) { failed++; }
    }
  }
  return failed == 0;
}

}  // namespace schedule
}  // namespace oneflow
