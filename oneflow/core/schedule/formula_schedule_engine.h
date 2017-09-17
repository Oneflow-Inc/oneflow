#ifndef FORMULA_SCHEDULE_ENGINE_H_
#define FORMULA_SCHEDULE_ENGINE_H_
#include "oneflow/core/schedule/longest_path_visitor.h"
#include "oneflow/core/schedule/schedule_engine.h"

namespace oneflow {
namespace schedule {

class FormulaScheduleEngine : public ScheduleEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FormulaScheduleEngine);
  explicit FormulaScheduleEngine(const Session& session)
      : ScheduleEngine(session) {}
  virtual ~FormulaScheduleEngine() = default;
  virtual std::unique_ptr<Schedule> StaticSchedule();
  virtual std::unique_ptr<Schedule> StaticSchedule(
      const std::function<uint32_t(uint64_t)>& get_regst_num);

 protected:
  virtual float GetSTaskWeight(const STask* task) const;
  virtual float EvaluateInitiationInterval() const;

 private:
  void ForEachRegstDescDuration(
      const std::function<void(const SRegstDesc*, float)>&) const;
  float GetRegstDescDuration(const LongestPathVisitor<const STask*>& lpath,
                             const SRegstDesc* regst_desc) const;
};

typedef FormulaScheduleEngine NaiveFormulaScheduleEngine;

}  // namespace schedule
}  // namespace oneflow

#endif  // FORMULA_SCHEDULE_ENGINE_H_
