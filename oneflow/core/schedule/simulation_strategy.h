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

  inline SimulatorScheduleEngine* schedule_engine() { return schedule_engine_; }

 protected:
  SimulatorScheduleEngine* schedule_engine_;
};

class EvaluationSimulationStrategy : public SimulationStrategy {
 public:
  explicit EvaluationSimulationStrategy(
      SimulatorScheduleEngine* schedule_engine)
      : SimulationStrategy(schedule_engine) {}
  ~EvaluationSimulationStrategy() = default;
  virtual float GetAscendentEndedAt(TaskInstance* instance);
  virtual void TimeLinePushBack(TaskInstance*, SDevice*) = 0;
  virtual void Retiming() = 0;
};

class EagerEvaluationStrategy : public EvaluationSimulationStrategy {
 public:
  explicit EagerEvaluationStrategy(SimulatorScheduleEngine* schedule_engine)
      : EvaluationSimulationStrategy(schedule_engine) {}
  virtual ~EagerEvaluationStrategy() = default;
  void TimeLinePushBack(TaskInstance* instance, SDevice* device) {}
  void Retiming() {}
};

class LazyEvaluationStrategy : public EvaluationSimulationStrategy {
 public:
  explicit LazyEvaluationStrategy(SimulatorScheduleEngine* schedule_engine)
      : EvaluationSimulationStrategy(schedule_engine) {}
  ~LazyEvaluationStrategy() = default;

  void TimeLinePushBack(TaskInstance* instance, SDevice* device);
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
  virtual std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>> Pick(
      const std::unordered_set<TaskArcInstance*>& tokens);
  virtual void BeforeRun(TaskInstance* instance) = 0;
  virtual void AfterRun(TaskInstance* instance) = 0;
  virtual void InitRegst(
      const std::function<uint32_t(uint64_t)>& get_regst_num) = 0;
  virtual float GetAscendentEndedAt(TaskInstance* instance);

  std::function<float(TaskInstance*)> get_ascendent_ended_at_;

 protected:
  void InitFuncs();
  virtual bool IsInstanceReady(TaskInstance* instance);

  std::function<TaskInstance*(TaskArcInstance*)> get_node_instance_;
  std::function<bool(TaskInstance*)> is_instance_ready_;
  std::function<SDevice*(TaskInstance*)> get_instance_device_;
  std::function<TaskInstance*(const std::list<TaskInstance*>&)>
      pick_instance_to_run_;
};

class UnlimitedMemoryStrategy : public MemorySimulationStrategy {
 public:
  UnlimitedMemoryStrategy(SimulatorScheduleEngine* schedule_engine)
      : MemorySimulationStrategy(schedule_engine) {}
  virtual void BeforeRun(TaskInstance* instance) {}
  virtual void AfterRun(TaskInstance* instance) {}
  void InitRegst(const std::function<uint32_t(uint64_t)>& get_regst_num) {}
};

class LimitedMemoryStrategy : public MemorySimulationStrategy {
 public:
  LimitedMemoryStrategy(SimulatorScheduleEngine* schedule_engine)
      : MemorySimulationStrategy(schedule_engine) {
    InitFuncIsInstanceReady();
  }
  void BeforeRun(TaskInstance* instance);
  void AfterRun(TaskInstance* instance);
  void InitRegst(const std::function<uint32_t(uint64_t)>& get_regst_num);

 private:
  void InitFuncIsInstanceReady();
  bool IsAllRegstDescReady(TaskInstance* instance);
  bool IsRegstDescReady(SRegstDesc* regst_desc, Batch* batch);
  SRegst* FindFreeRegst(SRegstDesc* regst_desc, Batch* batch);
  bool IsRegstFree(SRegst* regst);
  float RegstDescEndedAt(TaskInstance* instance);
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_
