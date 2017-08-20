/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_STATIC_SCHEDULER_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_STATIC_SCHEDULER_H_

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

#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class SimulatorScheduleEngine;

class SimulatorSchedule : public Schedule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SimulatorSchedule);
  explicit SimulatorSchedule(Session* session) : Schedule(session) {}
  void Clear();
  void UpdateTimeGapToLoss(SimulatorScheduleEngine* schedule_engine);
  void UpdateDuration(SimulatorScheduleEngine* schedule_engine);
  void UpdateRegstCount();
  void UpdateInterval(SimulatorScheduleEngine* schedule_engine);
  float GetDurationByTimeGapToLoss(TaskInstance* from, TaskInstance* to);
  void MergeTimeGapToLossInPlace(SimulatorSchedule* logger);

  void TimeLinePushBack(TaskInstance* instance, SDevice* device);
  void Retiming(SimulatorScheduleEngine* schedule_engine);
  void InitTimeNet(SimulatorScheduleEngine* schedule_engine);

  inline const NodeMgr<SRegst>& regst_node_mgr() const {
    return regst_node_mgr_;
  }
  inline NodeMgr<SRegst>& mut_regst_node_mgr() { return regst_node_mgr_; }
  inline const ArcMgr<Arc<TaskInstance, SRegst>>& regst_arc_mgr() const {
    return regst_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskInstance, SRegst>>& mut_regst_arc_mgr() {
    return regst_arc_mgr_;
  }
  inline const HasOneArcMgr<Arc<SRegst, SRegstDesc>>& r2rd_arc_mgr() const {
    return r2rd_arc_mgr_;
  }
  inline HasOneArcMgr<Arc<SRegst, SRegstDesc>>& mut_r2rd_arc_mgr() {
    return r2rd_arc_mgr_;
  }

  inline std::unordered_map<SRegst*, float>& mut_regst2ended_at() {
    return regst2ended_at_;
  }

  inline std::unordered_map<RegstDescInstance*, SRegst*>&
  mut_regst_desc_instance2regst() {
    return regst_desc_instance2regst_;
  }

  void ForeachNextTaskInstance(TaskInstance* task_instance,
                               const std::function<void(TaskInstance*)>& cb) {
    timenet_arc_mgr().Output(task_instance, cb);
  }

  void ForeachPrevTaskInstance(TaskInstance* task_instance,
                               const std::function<void(TaskInstance*)>& cb) {
    timenet_arc_mgr().Input(task_instance, cb);
  }

 protected:
  inline const ArcMgr<Arc<TaskInstance>>& timenet_arc_mgr() const {
    return timenet_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskInstance>>& mut_timenet_arc_mgr() {
    return timenet_arc_mgr_;
  }
  void WalkTimeNetReverse(SimulatorScheduleEngine* schedule_engine,
                          const std::function<void(TaskInstance*)>& cb);
  void WalkBpTimeNet(SimulatorScheduleEngine* schedule_engine,
                     const std::function<void(TaskInstance*)>& cb);

  std::unordered_map<SRegst*, float> regst2ended_at_;
  std::unordered_map<RegstDescInstance*, SRegst*> regst_desc_instance2regst_;
  NodeMgr<SRegst> regst_node_mgr_;
  ArcMgr<Arc<TaskInstance, SRegst>> regst_arc_mgr_;
  HasOneArcMgr<Arc<SRegst, SRegstDesc>> r2rd_arc_mgr_;

  ArcMgr<Arc<TaskInstance>> timenet_arc_mgr_;
  std::unordered_map<SDevice*, TaskInstance*> dev2current_instance_;
};

class SimulatorScheduleEngine : public ScheduleEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SimulatorScheduleEngine);
  SimulatorScheduleEngine(Session* session)
      : ScheduleEngine(session),
        schedule_(unique_ptr_new<SimulatorSchedule>(session)) {
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
    schedule_ = unique_ptr_new<SimulatorSchedule>(session());
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

  STask* EndNode() { return direction_->EndNode(); }

  inline float GetTime(float x) { return direction_->GetTime(x); }
  inline float GetStartTime(const std::pair<float, float>& p) {
    return direction_->GetStartTime(p);
  }
  inline float GetEndTime(const std::pair<float, float>& p) {
    return direction_->GetEndTime(p);
  }

  inline std::unordered_set<TaskArcInstance*>& mut_tokens() { return tokens_; }

  //	getter
  inline SimulatorSchedule* schedule() const { return schedule_.get(); }
  inline DirectionSimulationStrategy* direction() const {
    return direction_.get();
  }
  inline EvaluationSimulationStrategy* evaluation() const {
    return evaluation_.get();
  }
  inline MemorySimulationStrategy* memory() const { return memory_.get(); }

  friend class SimulatorSchedule;

 private:
  void InitStrategies();
  std::unique_ptr<Schedule> RunInTwoDirections(
      const std::function<uint32_t(uint64_t)>& get_regst_num);

  void SetStrategy(std::unique_ptr<DirectionSimulationStrategy>&& direction) {
    direction_ = std::move(direction);
  }
  void SetStrategy(std::unique_ptr<EvaluationSimulationStrategy>&& evaluation) {
    evaluation_ = std::move(evaluation);
  }
  void SetStrategy(std::unique_ptr<MemorySimulationStrategy>&& memory) {
    memory_ = std::move(memory);
  }

  void InitRegst(const std::function<uint32_t(uint64_t)>& get_regst_num) {
    memory_->InitRegst(get_regst_num);
  }

  inline void NewStartTokens() { return direction_->NewStartTokens(); }
  inline uint32_t NextArc(STask* node,
                          const std::function<void(TaskArc*)>& cb) {
    return direction_->NextArc(node, cb);
  }
  inline uint32_t PrevArc(STask* node,
                          const std::function<void(TaskArc*)>& cb) {
    return direction_->PrevArc(node, cb);
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
  std::unique_ptr<DirectionSimulationStrategy> direction_;
  std::unique_ptr<EvaluationSimulationStrategy> evaluation_;
  std::unique_ptr<MemorySimulationStrategy> memory_;
  std::unordered_set<TaskArcInstance*> tokens_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_STATIC_SCHEDULER_H_
