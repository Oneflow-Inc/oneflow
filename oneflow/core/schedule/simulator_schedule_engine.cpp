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
  auto graph = session()->graph();
  graph->arc_mgr().InputArc(graph->sink(), &arcs);
  auto batchs = session()->GetBatchNodes();
  session()->task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(graph->sink());
}

void SimulatorScheduleEngine::InitNodeBatchInstance(STask* node) {
  for (uint32_t i = 0; i < session()->nr_batch(); i++) {
    auto batch = session()->batch_node_mgr().Find(i);
    auto start_instance = session()->mut_task_instance_mgr().Find(batch, node);
    schedule()->mut_instance2ended_at()[start_instance] =
        std::make_pair(0u, 0u);
  }
}

void SimulatorScheduleEngine::NewSourceTokens() {
  ClearTmpData();
  std::list<TaskArc*> arcs;
  auto graph = session()->graph();
  graph->arc_mgr().OutputArc(graph->source(), &arcs);
  auto batchs = session()->GetBatchNodes();
  session()->task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(graph->source());
}

SDevice* SimulatorScheduleEngine::GetInstanceDevice(TaskInstance* instance) {
  SDevice* ret = nullptr;
  session()->graph()->device_arc_mgr().Output(instance->to(), &ret);
  return ret;
}

void SimulatorScheduleEngine::InitStrategies() {
  SetStrategy(unique_ptr_new<LazyEvaluationStrategy>(this));
  SetStrategy(unique_ptr_new<LimitedMemoryStrategy>(this));
}

bool SimulatorScheduleEngine::CompareInstanceOrder(TaskInstance* instance_a,
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
  SetStrategy(unique_ptr_new<LimitedMemoryStrategy>(this));
  return Run(get_regst_num);
}

std::unique_ptr<Schedule> SimulatorScheduleEngine::StaticSchedule() {
  SetStrategy(unique_ptr_new<UnlimitedMemoryStrategy>(this));
  return Run([](uint64_t) { return static_cast<uint32_t>(2u); });
}

std::unique_ptr<SimulatorSchedule> SimulatorScheduleEngine::Run(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  InitRegst(get_regst_num);
  NewSourceTokens();
  auto graph = session()->graph();
  while (mut_tokens().size()) {
    auto instances_picked = Pick(&mut_tokens());
    for (const auto& p : *instances_picked) {
      auto dev = dynamic_cast<SDevice*>(p.first);
      auto batch = p.second->from();
      BeforeRun(p.second);
      float ended_at = GetAscendentEndedAt(p.second);
      schedule()->mut_instance2ended_at()[p.second].first = ended_at;
      ended_at += (dev ? dev->time() : 0);
      schedule()->mut_device2ended_at()[p.first] = ended_at;
      schedule()->mut_instance2ended_at()[p.second].second = ended_at;
      TimeLinePushBack(p.second, dev);
      AfterRun(p.second);
      graph->arc_mgr().InputArc(p.second->to(), [&](TaskArc* arc) {
        auto instance_input =
            session()->task_arc_instance_mgr().Find(batch, arc);
        mut_tokens().erase(instance_input);
      });
      graph->arc_mgr().OutputArc(p.second->to(), [&](TaskArc* arc) {
        auto instance_output =
            session()->task_arc_instance_mgr().Find(batch, arc);
        mut_tokens().insert(instance_output);
      });
    }
    if (!instances_picked->size()) { break; }
  }
  schedule()->UpdateInterval(this);
  Retiming();
  schedule()->UpdateTimeGapToLoss(this);
  schedule()->UpdateDuration(this);
  schedule()->UpdateRegstCount();
  return GetSchedule();
}

}  // namespace schedule
}  // namespace oneflow
