#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

class SimulatorScheduleEngine;

class SimulationStrategy {
 public:
  SimulationStrategy(SimulatorScheduleEngine* schedule_engine)
      : schedule_engine_(schedule_engine) {}
  virtual ~SimulationStrategy() {}

  inline SimulatorScheduleEngine* schedule_engine() const {
    return schedule_engine_;
  }

 protected:
  SimulatorScheduleEngine* schedule_engine_;
};

class EvaluationSimulationStrategy : public SimulationStrategy {
 public:
  explicit EvaluationSimulationStrategy(
      SimulatorScheduleEngine* schedule_engine)
      : SimulationStrategy(schedule_engine) {}
  ~EvaluationSimulationStrategy() = default;
  virtual float GetAscendantEndedAt(const TaskInstance* instance) const;
  virtual void TimeLinePushBack(const TaskInstance*, const SDevice*) = 0;
  virtual void Retiming() = 0;
};

class EagerEvaluationStrategy : public EvaluationSimulationStrategy {
 public:
  explicit EagerEvaluationStrategy(SimulatorScheduleEngine* schedule_engine)
      : EvaluationSimulationStrategy(schedule_engine) {}
  virtual ~EagerEvaluationStrategy() = default;
  void TimeLinePushBack(const TaskInstance* instance, const SDevice* device) {}
  void Retiming() {}
};

class LazyEvaluationStrategy : public EvaluationSimulationStrategy {
 public:
  explicit LazyEvaluationStrategy(SimulatorScheduleEngine* schedule_engine)
      : EvaluationSimulationStrategy(schedule_engine) {}
  ~LazyEvaluationStrategy() = default;

  void TimeLinePushBack(const TaskInstance* instance, const SDevice* device);
  void Retiming();

 private:
  void InitTimeNet();
};

class MemorySimulationStrategy : public SimulationStrategy {
 public:
  MemorySimulationStrategy(SimulatorScheduleEngine* schedule_engine)
      : SimulationStrategy(schedule_engine) {
    InitFuncs();
  }
  virtual ~MemorySimulationStrategy() {}
  virtual std::unique_ptr<
      std::unordered_map<const SDevice*, const TaskInstance*>>
  Pick(const std::unordered_set<const TaskArcInstance*>& tokens) const;
  virtual void BeforeRun(const TaskInstance* instance, float time) = 0;
  virtual void AfterRun(const TaskInstance* instance, float time) = 0;
  virtual void InitRegst(
      const std::function<uint32_t(uint64_t)>& get_regst_num) = 0;
  virtual float GetAscendantEndedAt(const TaskInstance* instance) const;

  std::function<float(const TaskInstance*)> get_ascendant_ended_at_;

 protected:
  void InitFuncs();
  virtual bool IsInstanceReady(const TaskInstance* instance) const;

  std::function<const TaskInstance*(const TaskArcInstance*)> get_node_instance_;
  std::function<bool(const TaskInstance*)> is_instance_ready_;
  std::function<const SDevice*(const TaskInstance*)> get_instance_device_;
  std::function<const TaskInstance*(const std::list<const TaskInstance*>&)>
      pick_instance_to_run_;
};

class UnlimitedMemoryStrategy : public MemorySimulationStrategy {
 public:
  UnlimitedMemoryStrategy(SimulatorScheduleEngine* schedule_engine)
      : MemorySimulationStrategy(schedule_engine) {}
  virtual void BeforeRun(const TaskInstance* instance, float time) {}
  virtual void AfterRun(const TaskInstance* instance, float time) {}
  void InitRegst(const std::function<uint32_t(uint64_t)>& get_regst_num) {}
};

class LimitedMemoryStrategy : public MemorySimulationStrategy {
 public:
  LimitedMemoryStrategy(SimulatorScheduleEngine* schedule_engine)
      : MemorySimulationStrategy(schedule_engine) {
    InitFuncIsInstanceReady();
  }
  void BeforeRun(const TaskInstance* instance, float time);
  void AfterRun(const TaskInstance* instance, float time);
  void InitRegst(const std::function<uint32_t(uint64_t)>& get_regst_num);

 private:
  void InitFuncIsInstanceReady();
  bool IsAllRegstDescReady(const TaskInstance* instance) const;
  bool IsRegstDescReady(const SRegstDesc* regst_desc, const Batch* batch) const;
  const SRegst* FindFreeRegst(const SRegstDesc* regst_desc,
                              const Batch* batch) const;
  bool IsRegstFree(const SRegst* regst) const;
  float RegstDescEndedAt(const TaskInstance* instance) const;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_
