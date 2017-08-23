/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/bfs_visitor.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"

namespace oneflow {
namespace schedule {

void SimulatorScheduleEngine::ClearTmpData() {
  tokens_.clear();
  schedule()->Clear();
}

void SimulatorScheduleEngine::NewSinkTokens() {
  ClearTmpData();
  std::list<TaskArc*> arcs;
  const SGraph* graph = session()->graph();
  graph->arc_mgr().InputArc(graph->sink(), &arcs);
  auto batchs = session()->GetBatchNodes();
  session()->task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(graph->sink());
}

void SimulatorScheduleEngine::InitNodeBatchInstance(STask* node) {
  for (uint32_t i = 0; i < session()->nr_batch(); i++) {
    Batch* batch = session()->batch_node_mgr().Find(i);
    TaskInstance* start_instance =
        session()->task_instance_mgr().Find(batch, node);
    schedule()->mut_instance2ended_at()[start_instance] =
        std::make_pair(0.0, 0.0);
  }
}

void SimulatorScheduleEngine::NewSourceTokens() {
  ClearTmpData();
  std::list<TaskArc*> arcs;
  const SGraph* graph = session()->graph();
  graph->arc_mgr().OutputArc(graph->source(), &arcs);
  auto batchs = session()->GetBatchNodes();
  session()->task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(graph->source());
}

SDevice* SimulatorScheduleEngine::GetInstanceDevice(TaskInstance* instance) {
  SDevice* ret = nullptr;
  session()->graph()->device_arc_mgr().Output(instance->dst_node(), &ret);
  return ret;
}

void SimulatorScheduleEngine::InitStrategies() {
  SetStrategy(of_make_unique<LazyEvaluationStrategy>(this));
  SetStrategy(of_make_unique<LimitedMemoryStrategy>(this));
}

bool SimulatorScheduleEngine::CompareInstanceOrder(TaskInstance* instance_a,
                                                   TaskInstance* instance_b) {
  if (instance_a->dst_node() == instance_b->dst_node()) {
    // same node
    return instance_a->src_node()->id() < instance_b->src_node()->id();
  }
  if (instance_a->src_node() == instance_b->src_node()) {
    // same batch
    return instance_a->dst_node()->depth() > instance_b->dst_node()->depth();
  }
  return instance_a->dst_node()->depth() < instance_b->dst_node()->depth();
}

TaskInstance* SimulatorScheduleEngine::PickInstanceToRun(
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

std::unique_ptr<Schedule> SimulatorScheduleEngine::StaticSchedule(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  SetStrategy(of_make_unique<LimitedMemoryStrategy>(this));
  return Run(get_regst_num);
}

std::unique_ptr<Schedule> SimulatorScheduleEngine::StaticSchedule() {
  SetStrategy(of_make_unique<UnlimitedMemoryStrategy>(this));
  return Run([](uint64_t) { return static_cast<uint32_t>(2u); });
}

std::unique_ptr<SimulatorSchedule> SimulatorScheduleEngine::Run(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  InitRegst(get_regst_num);
  NewSourceTokens();
  const SGraph* graph = session()->graph();
  while (tokens().size()) {
    auto instances_picked = Pick(tokens());
    for (const auto& p : *instances_picked) {
      SDevice* dev = dynamic_cast<SDevice*>(p.first);
      Batch* batch = p.second->src_node();
      STask* task = p.second->dst_node();
      BeforeRun(p.second);
      float ended_at = GetAscendentEndedAt(p.second);
      schedule()->mut_instance2ended_at()[p.second].first = ended_at;
      ended_at += task->workload() * (dev ? dev->time() : 0.0);
      schedule()->mut_device2ended_at()[p.first] = ended_at;
      schedule()->mut_instance2ended_at()[p.second].second =
          ended_at + (dev ? dev->delay() : 0.0);
      TimeLinePushBack(p.second, dev);
      AfterRun(p.second);
      graph->arc_mgr().InputArc(p.second->dst_node(), [&](TaskArc* arc) {
        TaskArcInstance* instance_input =
            session()->task_arc_instance_mgr().Find(batch, arc);
        mut_tokens().erase(instance_input);
      });
      graph->arc_mgr().OutputArc(p.second->dst_node(), [&](TaskArc* arc) {
        TaskArcInstance* instance_output =
            session()->task_arc_instance_mgr().Find(batch, arc);
        mut_tokens().insert(instance_output);
      });
    }
    if (!instances_picked->size()) { break; }
  }
  schedule()->UpdateInterval();
  Retiming();
  schedule()->UpdateDuration();
  schedule()->UpdateRegstCount();
  return GetSchedule();
}

}  // namespace schedule
}  // namespace oneflow
