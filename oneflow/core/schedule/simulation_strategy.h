/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_

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
#include "oneflow/core/schedule/util.h"

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

class DirectionSimulationStrategy : public SimulationStrategy {
 public:
  DirectionSimulationStrategy(SimulatorScheduleEngine* schedule_engine)
      : SimulationStrategy(schedule_engine) {}
  virtual ~DirectionSimulationStrategy() {}

  virtual int32_t GetTime(int32_t x) = 0;
  virtual int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) = 0;
  virtual int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) = 0;
  virtual void NewStartTokens() = 0;
  virtual unsigned int NextArc(STask* node,
                               const std::function<void(TaskArc*)>& cb) = 0;
  virtual unsigned int Next(STask* node,
                            const std::function<void(STask*)>& cb) = 0;
  virtual unsigned int PrevArc(STask* node,
                               const std::function<void(TaskArc*)>& cb) = 0;
  virtual unsigned int Prev(STask* node,
                            const std::function<void(STask*)>& cb) = 0;
  virtual TaskInstance* GetNextNodeInstance(TaskArcInstance* arc) = 0;
  virtual bool CompareInstanceOrder(TaskInstance* instance_a,
                                    TaskInstance* instance_b) = 0;
  virtual TaskInstance* PickInstanceToRun(
      const std::list<TaskInstance*>& instances);
  virtual int HoldingRegstDesc(STask* node,
                               const std::function<void(SRegstDesc*)>& cb) = 0;
  virtual int RegstDescReleasingNode(SRegstDesc* regst_desc,
                                     const std::function<void(STask*)>& cb) = 0;
  virtual STask* StartNode() = 0;
  virtual STask* EndNode() = 0;
  virtual Batch* EndBatch() = 0;
  virtual uint32_t NextBatchId(uint32_t batch_id) = 0;
  virtual STask* GetFrom(TaskArc* arc) = 0;
  virtual STask* GetTo(TaskArc* arc) = 0;
};

class PositiveDirectionStrategy : public DirectionSimulationStrategy {
 public:
  PositiveDirectionStrategy(SimulatorScheduleEngine* schedule_engine)
      : DirectionSimulationStrategy(schedule_engine) {}
  virtual ~PositiveDirectionStrategy() {}
  int32_t GetTime(int32_t x) { return x; }
  int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.first);
  }
  int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.second);
  }
  void NewStartTokens();
  unsigned int NextArc(STask* node, const std::function<void(TaskArc*)>& cb);
  unsigned int Next(STask* node, const std::function<void(STask*)>& cb);
  unsigned int PrevArc(STask* node, const std::function<void(TaskArc*)>& cb);
  unsigned int Prev(STask* node, const std::function<void(STask*)>& cb);
  TaskInstance* GetNextNodeInstance(TaskArcInstance* arc);
  bool CompareInstanceOrder(TaskInstance* instance_a, TaskInstance* instance_b);
  int HoldingRegstDesc(STask* node, const std::function<void(SRegstDesc*)>& cb);
  int RegstDescReleasingNode(SRegstDesc* regst_desc,
                             const std::function<void(STask*)>& cb);
  STask* StartNode();
  STask* EndNode();
  Batch* EndBatch();
  STask* GetFrom(TaskArc* arc) { return arc->from(); }
  STask* GetTo(TaskArc* arc) { return arc->to(); }
  uint32_t NextBatchId(uint32_t batch_id) { return batch_id + 1; }
};

class NegativeDirectionStrategy : public DirectionSimulationStrategy {
 public:
  NegativeDirectionStrategy(SimulatorScheduleEngine* schedule_engine)
      : DirectionSimulationStrategy(schedule_engine) {}
  virtual ~NegativeDirectionStrategy() {}
  virtual int32_t GetTime(int32_t x) { return -x; }
  virtual int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.second);
  }
  virtual int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.first);
  }
  void NewStartTokens();
  unsigned int NextArc(STask* node, const std::function<void(TaskArc*)>& cb);
  unsigned int Next(STask* node, const std::function<void(STask*)>& cb);
  unsigned int PrevArc(STask* node, const std::function<void(TaskArc*)>& cb);
  unsigned int Prev(STask* node, const std::function<void(STask*)>& cb);
  TaskInstance* GetNextNodeInstance(TaskArcInstance* arc);
  bool CompareInstanceOrder(TaskInstance* instance_a, TaskInstance* instance_b);
  int HoldingRegstDesc(STask* node, const std::function<void(SRegstDesc*)>& cb);
  int RegstDescReleasingNode(SRegstDesc* regst_desc,
                             const std::function<void(STask*)>& cb);
  STask* StartNode();
  STask* EndNode();
  Batch* EndBatch();
  STask* GetFrom(TaskArc* arc) { return arc->to(); }
  STask* GetTo(TaskArc* arc) { return arc->from(); }
  uint32_t NextBatchId(uint32_t batch_id) { return batch_id - 1; }
};

class EvaluationSimulationStrategy : public SimulationStrategy {
 public:
  explicit EvaluationSimulationStrategy(
      SimulatorScheduleEngine* schedule_engine)
      : SimulationStrategy(schedule_engine) {}
  ~EvaluationSimulationStrategy() = default;
  virtual int32_t GetAscendentEndedAt(TaskInstance* instance);
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
      std::unordered_set<TaskArcInstance*>* tokens);
  virtual void BeforeRun(TaskInstance* instance) = 0;
  virtual void AfterRun(TaskInstance* instance) = 0;
  virtual void InitRegst(
      const std::function<uint32_t(uint64_t)>& get_regst_num) = 0;
  virtual int32_t GetAscendentEndedAt(TaskInstance* instance);

  std::function<int32_t(TaskInstance*)> get_ascendent_ended_at_;

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
  int32_t RegstDescEndedAt(TaskInstance* instance);
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATION_STRATEGY_H_
