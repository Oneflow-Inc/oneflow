/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_ENGINE_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_ENGINE_H_

#include <limits.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"
#include "oneflow/core/schedule/simulator_schedule.h"

namespace oneflow {
namespace schedule {

class SimulatorScheduleEngine : public ScheduleEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SimulatorScheduleEngine);
  SimulatorScheduleEngine(Session* session)
      : ScheduleEngine(session),
        schedule_(of_make_unique<SimulatorSchedule>(session)) {
    InitStrategies();
  }

  virtual ~SimulatorScheduleEngine() = default;

  SDevice* GetInstanceDevice(TaskInstance* instance);
  void NewSourceTokens();
  void NewSinkTokens();
  void ClearTmpData();
  void InitNodeBatchInstance(STask* node);

  std::unique_ptr<SimulatorSchedule> GetSchedule() {
    auto ret = std::move(schedule_);
    schedule_ = of_make_unique<SimulatorSchedule>(session());
    return ret;
  }

  std::unique_ptr<Schedule> StaticSchedule();
  std::unique_ptr<Schedule> StaticSchedule(
      const std::function<uint32_t(uint64_t)>& get_regst_num);
  std::unique_ptr<Schedule> StaticSchedule(uint32_t regst_max = 3u) {
    return StaticSchedule([=](uint64_t id) { return regst_max; });
  }

  std::unique_ptr<SimulatorSchedule> Run(
      const std::function<uint32_t(uint64_t)>& get_regst_num);

  TaskInstance* PickInstanceToRun(const std::list<TaskInstance*>& instances);
  bool CompareInstanceOrder(TaskInstance* instance_a, TaskInstance* instance_b);

  inline std::unordered_set<TaskArcInstance*>& mut_tokens() { return tokens_; }

  //	getter
  inline SimulatorSchedule* schedule() const { return schedule_.get(); }
  inline EvaluationSimulationStrategy* evaluation() const {
    return evaluation_.get();
  }
  inline MemorySimulationStrategy* memory() const { return memory_.get(); }

  friend class SimulatorSchedule;

 private:
  void InitStrategies();

  void SetStrategy(std::unique_ptr<EvaluationSimulationStrategy>&& evaluation) {
    evaluation_ = std::move(evaluation);
  }
  void SetStrategy(std::unique_ptr<MemorySimulationStrategy>&& memory) {
    memory_ = std::move(memory);
  }

  void InitRegst(const std::function<uint32_t(uint64_t)>& get_regst_num) {
    memory_->InitRegst(get_regst_num);
  }

  inline std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>> Pick(
      std::unordered_set<TaskArcInstance*>* tokens) {
    return memory_->Pick(tokens);
  }
  inline void TimeLinePushBack(TaskInstance* instance, SDevice* dev) {
    return evaluation_->TimeLinePushBack(instance, dev);
  }
  inline void Retiming() { return evaluation_->Retiming(); }
  inline void BeforeRun(TaskInstance* instance) {
    //    evaluation_->BeforeRun(instance);
    memory_->BeforeRun(instance);
  }
  inline void AfterRun(TaskInstance* instance) {
    //    evaluation_->AfterRun(instance);
    memory_->AfterRun(instance);
  }
  inline float GetAscendentEndedAt(TaskInstance* instance) {
    return memory_->get_ascendent_ended_at_(instance);
  }
  std::unique_ptr<SimulatorSchedule> schedule_;
  std::unique_ptr<EvaluationSimulationStrategy> evaluation_;
  std::unique_ptr<MemorySimulationStrategy> memory_;
  std::unordered_set<TaskArcInstance*> tokens_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_ENGINE_H_
