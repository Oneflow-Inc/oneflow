#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_ENGINE_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_ENGINE_H_

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
  SimulatorScheduleEngine(const Session& session)
      : ScheduleEngine(session),
        schedule_(of_make_unique<SimulatorSchedule>(session)) {
    InitStrategies();
  }

  virtual ~SimulatorScheduleEngine() = default;

  const SDevice* GetInstanceDevice(const TaskInstance* instance);
  void NewSourceTokens();
  void NewSinkTokens();
  void ClearTmpData();
  void InitNodeBatchInstance(const STask* node);

  std::unique_ptr<SimulatorSchedule> GetSchedule() {
    std::unique_ptr<SimulatorSchedule> ret = std::move(schedule_);
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

  const TaskInstance* PickInstanceToRun(
      const std::list<const TaskInstance*>& instances);
  bool CompareInstanceOrder(const TaskInstance* instance_a,
                            const TaskInstance* instance_b);

  inline const std::unordered_set<const TaskArcInstance*>& tokens() const {
    return tokens_;
  }
  inline std::unordered_set<const TaskArcInstance*>& mut_tokens() {
    return tokens_;
  }

  //	getter
  inline SimulatorSchedule* schedule() const { return schedule_.get(); }
  inline const EvaluationSimulationStrategy* evaluation() const {
    return evaluation_.get();
  }
  inline const MemorySimulationStrategy* memory() const {
    return memory_.get();
  }

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

  inline std::unique_ptr<
      std::unordered_map<const SDevice*, const TaskInstance*>>
  Pick(const std::unordered_set<const TaskArcInstance*>& tokens) {
    return memory_->Pick(tokens);
  }
  inline void TimeLinePushBack(const TaskInstance* instance,
                               const SDevice* dev) {
    return evaluation_->TimeLinePushBack(instance, dev);
  }
  inline void Retiming() { return evaluation_->Retiming(); }
  inline void BeforeRun(const TaskInstance* instance, float time) {
    const_cast<MemorySimulationStrategy*>(memory_.get())
        ->BeforeRun(instance, time);
  }
  inline void AfterRun(const TaskInstance* instance, float time) {
    const_cast<MemorySimulationStrategy*>(memory_.get())
        ->AfterRun(instance, time);
  }
  inline float GetAscendantEndedAt(const TaskInstance* instance) {
    return memory_->get_ascendant_ended_at_(instance);
  }
  std::unique_ptr<SimulatorSchedule> schedule_;
  std::unique_ptr<EvaluationSimulationStrategy> evaluation_;
  std::unique_ptr<MemorySimulationStrategy> memory_;
  std::unordered_set<const TaskArcInstance*> tokens_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_ENGINE_H_
