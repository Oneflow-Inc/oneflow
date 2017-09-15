#ifndef ONEFLOW_CORE_SCHEDULE_UGRAPH_SCHEDULE_ENGINE_H_
#define ONEFLOW_CORE_SCHEDULE_UGRAPH_SCHEDULE_ENGINE_H_

#include "oneflow/core/schedule/formula_schedule_engine.h"

namespace oneflow {
namespace schedule {

class UGraphScheduleEngine : public FormulaScheduleEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UGraphScheduleEngine);
  explicit UGraphScheduleEngine(const Session& session)
      : FormulaScheduleEngine(session) {
    Init();
  }
  virtual ~UGraphScheduleEngine() = default;

 protected:
  virtual float GetSTaskWeight(STask* task) const override;
  virtual float EvaluateInitiationInterval() const override;
  void Init();
  void InitTaskWeight();
  void InitII();

 private:
  std::unordered_map<STask*, float> task2weight_;
  float initiation_interval_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_UGRAPH_SCHEDULE_ENGINE_H_
