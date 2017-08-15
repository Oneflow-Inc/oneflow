/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/simulation_strategy.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"

namespace oneflow {
namespace schedule {

void LazyEvaluationStrategy::TimeLinePushBack(TaskInstance* instance,
                                              SDevice* device) {
  schedule_engine()->schedule()->TimeLinePushBack(instance, device);
}

int PositiveDirectionStrategy::HoldingRegstDesc(
    STask* node, const std::function<void(SRegstDesc*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->produced_regst_desc_mgr().Output(node, cb);
}

int PositiveDirectionStrategy::RegstDescReleasingNode(
    SRegstDesc* regst_desc, const std::function<void(STask*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->subscribed_regst_desc_mgr().Input(regst_desc, cb);
}

int NegativeDirectionStrategy::HoldingRegstDesc(
    STask* node, const std::function<void(SRegstDesc*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->subscribed_regst_desc_mgr().Output(node, cb);
}

int NegativeDirectionStrategy::RegstDescReleasingNode(
    SRegstDesc* regst_desc, const std::function<void(STask*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->produced_regst_desc_mgr().Input(regst_desc, cb);
}

STask* PositiveDirectionStrategy::StartNode() {
  return schedule_engine()->session()->graph()->source();
}

STask* PositiveDirectionStrategy::EndNode() {
  return schedule_engine()->session()->graph()->sink();
}

Batch* PositiveDirectionStrategy::EndBatch() {
  auto session = schedule_engine()->session();
  return session->batch_node_mgr().Find(session->nr_batch() - 1);
}

bool PositiveDirectionStrategy::CompareInstanceOrder(TaskInstance* instance_a,
                                                     TaskInstance* instance_b) {
  if (instance_a->to() == instance_b->to()) {
    // same node
    return instance_a->from()->id() < instance_b->from()->id();
  }
  if (instance_a->from() == instance_b->from()) {
    // same batch
    return instance_a->to()->depth() > instance_b->to()->depth();
  }
  return instance_a->to()->depth() < instance_b->to()->depth();
}

STask* NegativeDirectionStrategy::StartNode() {
  return schedule_engine()->session()->graph()->sink();
}

STask* NegativeDirectionStrategy::EndNode() {
  return schedule_engine()->session()->graph()->source();
}

Batch* NegativeDirectionStrategy::EndBatch() {
  return schedule_engine()->session()->batch_node_mgr().Find(0u);
}

bool NegativeDirectionStrategy::CompareInstanceOrder(TaskInstance* instance_a,
                                                     TaskInstance* instance_b) {
  if (instance_a->to() == instance_b->to()) {
    // same node
    return instance_a->from()->id() > instance_b->from()->id();
  }
  if (instance_a->from() == instance_b->from()) {
    // same batch
    return instance_a->to()->depth() < instance_b->to()->depth();
  }
  return instance_a->to()->depth() > instance_b->to()->depth();
}

TaskInstance* DirectionSimulationStrategy::PickInstanceToRun(
    const std::list<TaskInstance*>& instances) {
  TaskInstance* ret = nullptr;
  if (instances.size()) {
    auto itt = instances.begin();
    ret = *itt;
    for (; itt != instances.end(); itt++) {
      if (CompareInstanceOrder(*itt, ret)) { ret = *itt; }
    }
  }
  return ret;
}

void MemorySimulationStrategy::InitFuncs() {
  get_node_instance_ = [&](TaskArcInstance* arc) {
    auto direction = schedule_engine()->direction();
    return direction->GetNextNodeInstance(arc);
  };
  is_instance_ready_ = std::bind(&MemorySimulationStrategy::IsInstanceReady,
                                 this, std::placeholders::_1);
  get_instance_device_ = std::bind(&SimulatorScheduleEngine::GetInstanceDevice,
                                   schedule_engine(), std::placeholders::_1);
  get_ascendent_ended_at_ =
      std::bind(&MemorySimulationStrategy::GetAscendentEndedAt, this,
                std::placeholders::_1);
  pick_instance_to_run_ = [&](const std::list<TaskInstance*>& instances) {
    auto direction = schedule_engine()->direction();
    return direction->PickInstanceToRun(instances);
  };
}

TaskInstance* NegativeDirectionStrategy::GetNextNodeInstance(
    TaskArcInstance* arc) {
  auto session = schedule_engine()->session();
  return session->task_instance_mgr().Find(arc->from(), arc->to()->from());
}

TaskInstance* PositiveDirectionStrategy::GetNextNodeInstance(
    TaskArcInstance* arc) {
  auto session = schedule_engine()->session();
  return session->task_instance_mgr().Find(arc->from(), arc->to()->to());
}

void PositiveDirectionStrategy::NewStartTokens() {
  schedule_engine()->NewSourceTokens();
}

bool MemorySimulationStrategy::IsInstanceReady(TaskInstance* instance) {
  bool ready = true;
  auto session = schedule_engine()->session();
  auto direction = schedule_engine()->direction();
  direction->PrevArc(instance->to(), [&](TaskArc* arc) {
    auto instance_input =
        session->task_arc_instance_mgr().Find(instance->from(), arc);
    if (schedule_engine()->mut_tokens().find(instance_input)
        == schedule_engine()->mut_tokens().end()) {
      ready = false;
    }
  });
  return ready;
}

void NegativeDirectionStrategy::NewStartTokens() {
  schedule_engine()->NewSinkTokens();
}

unsigned int PositiveDirectionStrategy::PrevArc(
    STask* node, const std::function<void(TaskArc*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().InputArc(node, cb);
}

unsigned int PositiveDirectionStrategy::Prev(
    STask* node, const std::function<void(STask*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().Input(node, cb);
}

unsigned int PositiveDirectionStrategy::NextArc(
    STask* node, const std::function<void(TaskArc*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().OutputArc(node, cb);
}

unsigned int PositiveDirectionStrategy::Next(
    STask* node, const std::function<void(STask*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().Output(node, cb);
}

unsigned int NegativeDirectionStrategy::PrevArc(
    STask* node, const std::function<void(TaskArc*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().OutputArc(node, cb);
}

unsigned int NegativeDirectionStrategy::Prev(
    STask* node, const std::function<void(STask*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().Output(node, cb);
}

unsigned int NegativeDirectionStrategy::NextArc(
    STask* node, const std::function<void(TaskArc*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().InputArc(node, cb);
}

unsigned int NegativeDirectionStrategy::Next(
    STask* node, const std::function<void(STask*)>& cb) {
  auto graph = schedule_engine()->session()->graph();
  return graph->arc_mgr().Input(node, cb);
}

void LimitedMemoryStrategy::InitFuncIsInstanceReady() {
  is_instance_ready_ = [&](TaskInstance* instance) {
    return IsInstanceReady(instance) && IsAllRegstDescReady(instance);
  };
  get_ascendent_ended_at_ = [&](TaskInstance* instance) {
    auto evaluation = schedule_engine()->evaluation();
    return std::max(evaluation->GetAscendentEndedAt(instance),
                    RegstDescEndedAt(instance));
  };
}

void LazyEvaluationStrategy::Retiming() {
  return schedule_engine()->schedule()->Retiming(schedule_engine());
}

void LazyEvaluationStrategy::InitTimeNet() {
  return schedule_engine()->schedule()->InitTimeNet(schedule_engine());
}

void LimitedMemoryStrategy::InitRegst(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  auto session = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  session->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    auto count = get_regst_num(regst_desc->id());
    for (uint32_t i = 0; i < count; i++) {
      auto regst = schedule->mut_regst_node_mgr().Create(
          std::to_string(regst_desc->id()));
      schedule->mut_r2rd_arc_mgr().CreateIfNotFound(regst, regst_desc);
    }
  });
}

int32_t EvaluationSimulationStrategy::GetAscendentEndedAt(
    TaskInstance* instance) {
  int32_t ended_at = 0;
  auto session = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto direction = schedule_engine()->direction();
  direction->Prev(instance->to(), [&](STask* node) {
    auto instance_input =
        session->task_instance_mgr().Find(instance->from(), node);
    auto itt = schedule->instance2ended_at().find(instance_input);
    auto token_ended_at = INT_MAX;
    if (itt != schedule->instance2ended_at().end()) {
      token_ended_at = itt->second.second;
    }
    ended_at = std::max(ended_at, token_ended_at);
  });
  auto dev = schedule_engine()->GetInstanceDevice(instance);
  return std::max(ended_at, schedule->mut_device2ended_at()[dev]);
}

int32_t MemorySimulationStrategy::GetAscendentEndedAt(TaskInstance* instance) {
  auto evaluation = schedule_engine()->evaluation();
  return evaluation->GetAscendentEndedAt(instance);
}

int32_t LimitedMemoryStrategy::RegstDescEndedAt(TaskInstance* instance) {
  int32_t ended_at = 0;
  auto schedule = schedule_engine()->schedule();
  auto direction = schedule_engine()->direction();
  direction->HoldingRegstDesc(instance->to(), [&](SRegstDesc* regst_desc) {
    auto regst = FindFreeRegst(regst_desc, instance->from());
    ended_at = std::max(ended_at, schedule->mut_regst2ended_at()[regst]);
  });
  return ended_at;
}

void LimitedMemoryStrategy::BeforeRun(TaskInstance* instance) {
  auto session = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto direction = schedule_engine()->direction();
  direction->HoldingRegstDesc(instance->to(), [&](SRegstDesc* regst_desc) {
    auto regst = FindFreeRegst(regst_desc, instance->from());
    auto regst_desc_instance =
        session->regst_desc_instance_mgr().Find(instance->from(), regst_desc);
    if (!regst) {
      // BUG
      return;
    }
    schedule->mut_regst_desc_instance2regst()[regst_desc_instance] = regst;
    direction->RegstDescReleasingNode(regst_desc, [&](STask* node) {
      TaskInstance* subscriber_instance =
          session->task_instance_mgr().Find(instance->from(), node);
      schedule->mut_regst_arc_mgr().CreateIfNotFound(subscriber_instance,
                                                     regst);
    });
  });
}

void LimitedMemoryStrategy::AfterRun(TaskInstance* instance) {
  std::list<Arc<TaskInstance, SRegst>*> occupied_arcs;
  auto schedule = schedule_engine()->schedule();
  schedule->regst_arc_mgr().OutputArc(instance, &occupied_arcs);
  for (auto arc : occupied_arcs) {
    schedule->mut_regst2ended_at()[arc->to()] =
        schedule->mut_instance2ended_at()[instance].second;
    schedule->mut_regst_arc_mgr().Delete(arc->id());
  }
}

bool LimitedMemoryStrategy::IsAllRegstDescReady(TaskInstance* instance) {
  bool all_ready = true;
  auto direction = schedule_engine()->direction();
  direction->HoldingRegstDesc(instance->to(), [&](SRegstDesc* regst_desc) {
    all_ready = (all_ready && IsRegstDescReady(regst_desc, instance->from()));
  });
  return all_ready;
}

bool LimitedMemoryStrategy::IsRegstFree(SRegst* regst) {
  auto schedule = schedule_engine()->schedule();
  return schedule->regst_arc_mgr().Input(regst) == 0;
}

bool LimitedMemoryStrategy::IsRegstDescReady(SRegstDesc* regst_desc,
                                             Batch* batch) {
  auto sess = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto regst_desc_instance =
      sess->regst_desc_instance_mgr().Find(batch, regst_desc);
  bool free = schedule->mut_regst_desc_instance2regst()[regst_desc_instance];
  if (!free) {
    schedule->r2rd_arc_mgr().Input(regst_desc, [&](SRegst* regst) {
      free = (free || IsRegstFree(regst));
    });
  }
  return free;
}

SRegst* LimitedMemoryStrategy::FindFreeRegst(SRegstDesc* regst_desc,
                                             Batch* batch) {
  auto sess = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto regst_desc_instance =
      sess->regst_desc_instance_mgr().Find(batch, regst_desc);
  SRegst* ret = schedule->mut_regst_desc_instance2regst()[regst_desc_instance];
  if (!ret) {
    int32_t ended_at = INT_MAX;
    schedule->r2rd_arc_mgr().Input(regst_desc, [&](SRegst* regst) {
      if (IsRegstFree(regst)) {
        if (schedule->mut_regst2ended_at()[regst] < ended_at) {
          // first recycled register
          ended_at = schedule->mut_regst2ended_at()[regst];
          ret = regst;
        }
      }
    });
  }
  return ret;
}

std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>>
MemorySimulationStrategy::Pick(std::unordered_set<TaskArcInstance*>* tokens) {
  auto all_instances = XDistinct<TaskInstance*>(*tokens, get_node_instance_);
  auto ready_instances =
      XFilter<TaskInstance*>(*all_instances, is_instance_ready_);
  auto instances_groupby_ended_at =
      XGroupBy<int32_t>(*ready_instances, get_ascendent_ended_at_);
  auto first_finished = XAssocKMin(*instances_groupby_ended_at);
  auto instances_groupby_dev =
      XGroupBy<SDevice*>(first_finished->second, get_instance_device_);
  auto instances_picked =
      XAssocVMap<TaskInstance*>(*instances_groupby_dev, pick_instance_to_run_);
  return instances_picked;
}

}  // namespace schedule
}  // namespace oneflow
