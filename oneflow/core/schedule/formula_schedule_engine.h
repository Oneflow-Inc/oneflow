#ifndef FORMULA_SCHEDULE_ENGINE_H_
#define FORMULA_SCHEDULE_ENGINE_H_
#include "oneflow/core/schedule/schedule_engine.h"

namespace oneflow {
namespace schedule {

class FormulaScheduleEngine : public ScheduleEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FormulaScheduleEngine);
  FormulaScheduleEngine(const Session* session) : ScheduleEngine(session) {}
  ~FormulaScheduleEngine() = default;
  std::unique_ptr<Schedule> StaticSchedule();
  std::unique_ptr<Schedule> StaticSchedule(
      const std::function<uint32_t(uint64_t)>& get_regst_num);
};

}  // namespace schedule
}  // namespace oneflow

#endif  // FORMULA_SCHEDULE_ENGINE_H_
