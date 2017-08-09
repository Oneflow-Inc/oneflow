/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_IMPLEMENT_SIMULATOR_H_
#define ONEFLOW_CORE_SCHEDULE_IMPLEMENT_SIMULATOR_H_

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

#include "oneflow/core/schedule/node.h"
#include "oneflow/core/schedule/policy.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class SimulatorSession;
class Mode;

class SessionLogger : public ScheduleResult {
 public:
  SessionLogger() : ScheduleResult() {}
  DEFINE_METHOD_TYPE();
  void Clear();
  void UpdateTimeGapToLoss(SimulatorSession* session, Mode* strategy);
  void UpdateDuration(SimulatorSession* session, Mode* strategy);
  void UpdateInterval(SimulatorSession* session, Mode* strategy);
  void MergeTimeGapToLossInPlace(SessionLogger* logger);
  float GetDurationByTimeGapToLoss(TaskInstance* from, TaskInstance* to);
};

class SimulatorSession : public Session {
 public:
  explicit SimulatorSession(SGraph* graph, uint32_t nr_batch = 2u)
      : Session(graph, nr_batch), logger_(unique_ptr_new<SessionLogger>()) {}

  SDevice* GetInstanceDevice(TaskInstance* instance);

  void NewSourceTokens();
  void NewSinkTokens();
  void ClearTmpData();
  void InitNodeBatchInstance(STask* node);

  SessionLogger* logger() { return logger_.get(); }
  std::unique_ptr<SessionLogger>& mut_logger() { return logger_; }
  std::unique_ptr<SessionLogger> GetLoggerThenReset() {
    auto ret = std::move(logger_);
    logger_ = unique_ptr_new<SessionLogger>();
    return ret;
  }

  std::unordered_set<TaskArcInstance*> tokens_;
  std::unique_ptr<SessionLogger> logger_;

  //  struct PipeSpec {
  //    float duration;
  //    float freq;
  //    uint32_t count;
  //  };
  //  typedef std::unordered_map<uint32_t, PipeSpec> PipeCount;
};

class Strategy {
 public:
  Strategy(SimulatorSession* sess) : sess_(sess) {}
  virtual ~Strategy() {}

  inline SimulatorSession* Sess() { return sess_; }

 protected:
  SimulatorSession* sess_;
};

class DirectionStrategy : public Strategy {
 public:
  DirectionStrategy(SimulatorSession* sess) : Strategy(sess) {}
  virtual ~DirectionStrategy() {}

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

class PositiveStrategy : public DirectionStrategy {
 public:
  PositiveStrategy(SimulatorSession* sess) : DirectionStrategy(sess) {}
  virtual ~PositiveStrategy() {}
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
  STask* StartNode() { return Sess()->graph()->source(); }
  STask* EndNode() { return Sess()->graph()->sink(); }
  Batch* EndBatch() {
    return Sess()->batch_node_mgr().Find(Sess()->nr_batch() - 1);
  }
  STask* GetFrom(TaskArc* arc) { return arc->from(); }
  STask* GetTo(TaskArc* arc) { return arc->to(); }
  uint32_t NextBatchId(uint32_t batch_id) { return batch_id + 1; }
};

class NegativeStrategy : public DirectionStrategy {
 public:
  NegativeStrategy(SimulatorSession* sess) : DirectionStrategy(sess) {}
  virtual ~NegativeStrategy() {}
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
  STask* StartNode() { return Sess()->graph()->sink(); }
  STask* EndNode() { return Sess()->graph()->source(); }
  Batch* EndBatch() { return Sess()->batch_node_mgr().Find(0u); }
  STask* GetFrom(TaskArc* arc) { return arc->to(); }
  STask* GetTo(TaskArc* arc) { return arc->from(); }
  uint32_t NextBatchId(uint32_t batch_id) { return batch_id - 1; }
};

class EvaluationStrategy : public Strategy {
 public:
  EvaluationStrategy(DirectionStrategy* direction)
      : Strategy(direction->Sess()), direction_(direction) {}
  virtual int32_t GetAscendentEndedAt(TaskInstance* instance);
  virtual void TimeLinePushBack(TaskInstance*, SDevice*) = 0;
  virtual void Retiming() = 0;

 protected:
  DirectionStrategy* direction_;
};

class EagerStrategy : public EvaluationStrategy {
 public:
  EagerStrategy(DirectionStrategy* direction) : EvaluationStrategy(direction) {}
  void TimeLinePushBack(TaskInstance* instance, SDevice* device) {}
  void Retiming(){};
};

class LazyStrategy : public EvaluationStrategy {
 public:
  LazyStrategy(DirectionStrategy* direction) : EvaluationStrategy(direction) {
    InitTimeNet();
  }

  void TimeLinePushBack(TaskInstance* instance, SDevice* device);
  void Retiming();

 protected:
  inline const ArcMgr<Arc<TaskInstance>>& timenet_arc_mgr() const {
    return timenet_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskInstance>>& mut_timenet_arc_mgr() {
    return timenet_arc_mgr_;
  }
  void InitTimeNet();
  void WalkTimeNetReverse(const std::function<void(TaskInstance*)>& cb);

  ArcMgr<Arc<TaskInstance>> timenet_arc_mgr_;
  std::unordered_map<SDevice*, TaskInstance*> dev2current_instance_;
};

class ResourceStrategy : public Strategy {
 public:
  ResourceStrategy(DirectionStrategy* direction, EvaluationStrategy* evaluation)
      : Strategy(direction->Sess()),
        evaluation_(evaluation),
        direction_(direction) {
    InitFuncs();
  }
  virtual ~ResourceStrategy() {}
  virtual std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>> Pick(
      std::unordered_set<TaskArcInstance*>* tokens);
  virtual void BeforeRun(TaskInstance* instance) = 0;
  virtual void AfterRun(TaskInstance* instance) = 0;
  virtual int32_t GetAscendentEndedAt(TaskInstance* instance);

  std::function<int32_t(TaskInstance*)> get_ascendent_ended_at_;

 protected:
  void InitFuncs();
  virtual bool IsInstanceReady(TaskInstance* instance);

  std::function<TaskInstance*(TaskArcInstance*)> get_node_instance_;
  std::function<bool(TaskInstance*)> is_instance_ready_;
  std::function<SDevice*(TaskInstance*)> get_instance_device_;
  EvaluationStrategy* evaluation_;
  DirectionStrategy* direction_;
  std::function<TaskInstance*(const std::list<TaskInstance*>&)>
      pick_instance_to_run_;
};

class UnlimitedStrategy : public ResourceStrategy {
 public:
  UnlimitedStrategy(DirectionStrategy* direction, EvaluationStrategy* evalution)
      : ResourceStrategy(direction, evalution) {}
  virtual void BeforeRun(TaskInstance* instance) {}
  virtual void AfterRun(TaskInstance* instance) {}
};

class LimitedStrategy : public ResourceStrategy {
 public:
  LimitedStrategy(DirectionStrategy* direction, EvaluationStrategy* evaluation,
                  const std::function<uint64_t(uint32_t)>& get_regst_num)
      : ResourceStrategy(direction, evaluation) {
    InitRegst(get_regst_num);
    InitFuncIsInstanceReady();
  }
  void BeforeRun(TaskInstance* instance);
  void AfterRun(TaskInstance* instance);

 private:
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

  void InitRegst(const std::function<uint64_t(uint32_t)>& get_regst_num);
  void InitFuncIsInstanceReady();
  bool IsAllRegstDescReady(TaskInstance* instance);
  bool IsRegstDescReady(SRegstDesc* regst_desc, Batch* batch);
  SRegst* FindFreeRegst(SRegstDesc* regst_desc, Batch* batch);
  bool IsRegstFree(SRegst* regst);
  int32_t RegstDescEndedAt(TaskInstance* instance);
  std::unordered_map<SRegst*, int32_t> regst2ended_at_;
  std::unordered_map<RegstDescInstance*, SRegst*> regst_desc_instance2regst_;
  NodeMgr<SRegst> regst_node_mgr_;
  ArcMgr<Arc<TaskInstance, SRegst>> regst_arc_mgr_;
  HasOneArcMgr<Arc<SRegst, SRegstDesc>> r2rd_arc_mgr_;
};

class Mode : public Strategy {
 public:
  Mode(SimulatorSession* sess) : Strategy(sess) {}
  virtual ~Mode() {}
  DEFINE_PURE_VIRTUAL_TYPE();
  inline int32_t GetTime(int32_t x) { return direction_->GetTime(x); }
  inline int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) {
    return direction_->GetStartTime(p);
  }
  inline int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) {
    return direction_->GetEndTime(p);
  }
  void Run();

 protected:
  inline void NewStartTokens() { return direction_->NewStartTokens(); }
  inline unsigned int NextArc(STask* node,
                              const std::function<void(TaskArc*)>& cb) {
    return direction_->NextArc(node, cb);
  }
  inline unsigned int PrevArc(STask* node,
                              const std::function<void(TaskArc*)>& cb) {
    return direction_->PrevArc(node, cb);
  }
  inline std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>> Pick(
      std::unordered_set<TaskArcInstance*>* tokens) {
    return resource_->Pick(tokens);
  }
  inline void TimeLinePushBack(TaskInstance* instance, SDevice* dev) {
    return evaluation_->TimeLinePushBack(instance, dev);
  }
  inline void Retiming() { return evaluation_->Retiming(); }
  inline void BeforeRun(TaskInstance* instance) {
    //    evaluation_->BeforeRun(instance);
    resource_->BeforeRun(instance);
  }
  inline void AfterRun(TaskInstance* instance) {
    //    evaluation_->AfterRun(instance);
    resource_->AfterRun(instance);
  }
  inline int32_t GetAscendentEndedAt(TaskInstance* instance) {
    return resource_->get_ascendent_ended_at_(instance);
  }
  void SetStrategies(std::unique_ptr<DirectionStrategy>&& direction,
                     std::unique_ptr<EvaluationStrategy>&& evaluation,
                     std::unique_ptr<ResourceStrategy>&& resource) {
    direction_ = std::move(direction);
    evaluation_ = std::move(evaluation);
    resource_ = std::move(resource);
  }
  std::unique_ptr<DirectionStrategy> direction_;
  std::unique_ptr<EvaluationStrategy> evaluation_;
  std::unique_ptr<ResourceStrategy> resource_;
};

template<typename DirectionStrategyType,
         typename EvaluationStrategyType = LazyStrategy>
class UnlimitedMode : public Mode {
 public:
  explicit UnlimitedMode(SimulatorSession* sess) : Mode(sess) {
    auto direction = unique_ptr_new<DirectionStrategyType>(sess);
    auto evaluation = unique_ptr_new<EvaluationStrategyType>(&*direction);
    auto resource =
        unique_ptr_new<UnlimitedStrategy>(&*direction, &*evaluation);
    SetStrategies(std::move(direction), std::move(evaluation),
                  std::move(resource));
  }
  DEFINE_METHOD_TYPE();
};

template<typename DirectionStrategyType,
         typename EvaluationStrategyType = EagerStrategy>
class LimitedMode : public Mode {
 public:
  LimitedMode(SimulatorSession* sess,
              const std::function<uint64_t(uint32_t)>& get_regst_num)
      : Mode(sess) {
    auto direction = unique_ptr_new<DirectionStrategyType>(sess);
    auto evaluation = unique_ptr_new<EvaluationStrategyType>(&*direction);
    auto resource = unique_ptr_new<LimitedStrategy>(&*direction, &*evaluation,
                                                    get_regst_num);
    SetStrategies(std::move(direction), std::move(evaluation),
                  std::move(resource));
  }
  DEFINE_METHOD_TYPE();
};

class StaticSchedulerSimulatorPolicy : public StaticSchedulerPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(StaticSchedulerSimulatorPolicy,
                               StaticSchedulerPolicy);
  virtual std::unique_ptr<Session> MakeSession(const SGraph& graph);
  virtual std::unique_ptr<ScheduleResult> Schedule(const Session& session);
};

class RetimingSimulatorPolicy : public RetimingPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(RetimingSimulatorPolicy, RetimingPolicy);

  virtual void Retiming(const Session& session, ScheduleResult* result);
};

class AllocatorSimulatorPolicy : public AllocatorPolicy {
 public:
  POLICY_IMPLEMENT_BOILERPLATE(AllocatorSimulatorPolicy, AllocatorPolicy);

  virtual void AllocateFromSchedule(const Session& session,
                                    ScheduleResult* result);
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_IMPLEMENT_SIMULATOR_H_
