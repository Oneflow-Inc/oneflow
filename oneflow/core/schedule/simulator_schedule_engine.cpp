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
  std::list<const TaskArc*> arcs;
  sgraph().arc_mgr().InputArc(sgraph().sink(), &arcs);
  auto batchs = session().GetBatchNodes();
  session().task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](const TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(sgraph().sink());
}

void SimulatorScheduleEngine::InitNodeBatchInstance(const STask* node) {
  for (uint32_t i = 0; i < session().nr_batch(); i++) {
    const Batch* batch = session().batch_node_mgr().Find(i);
    const TaskInstance* start_instance =
        session().task_instance_mgr().Find(batch, node);
    schedule()->mut_instance2ended_at()[start_instance] =
        std::make_pair(0.0, 0.0);
  }
}

void SimulatorScheduleEngine::NewSourceTokens() {
  ClearTmpData();
  std::list<const TaskArc*> arcs;
  sgraph().arc_mgr().OutputArc(sgraph().source(), &arcs);
  auto batchs = session().GetBatchNodes();
  session().task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](const TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(sgraph().source());
}

const SDevice* SimulatorScheduleEngine::GetInstanceDevice(
    const TaskInstance* instance) {
  const SDevice* ret = nullptr;
  sgraph().device_arc_mgr().Output(instance->dst_node(), &ret);
  return ret;
}

void SimulatorScheduleEngine::InitStrategies() {
  SetStrategy(of_make_unique<LazyEvaluationStrategy>(this));
  SetStrategy(of_make_unique<LimitedMemoryStrategy>(this));
}

bool SimulatorScheduleEngine::CompareInstanceOrder(
    const TaskInstance* instance_a, const TaskInstance* instance_b) {
  if (instance_a->src_node()->id() < instance_b->src_node()->id()) return true;
  return (instance_a->src_node() == instance_b->src_node())
         && (instance_a->dst_node()->depth() > instance_b->dst_node()->depth());
}

const TaskInstance* SimulatorScheduleEngine::PickInstanceToRun(
    const std::list<const TaskInstance*>& instances) {
  const TaskInstance* ret = nullptr;
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
  while (tokens().size()) {
    auto instances_picked = Pick(tokens());
    for (const auto& p : *instances_picked) {
      const SDevice* dev = p.first;
      const Batch* batch = p.second->src_node();
      const STask* task = p.second->dst_node();
      float ended_at = GetAscendantEndedAt(p.second);
      BeforeRun(p.second, ended_at);
      schedule()->mut_instance2ended_at()[p.second].first = ended_at;
      ended_at += task->workload() * (dev ? dev->time() : 0.0);
      schedule()->mut_device2ended_at()[p.first] = ended_at;
      ended_at = ended_at + (dev ? dev->delay() : 0.0);
      schedule()->mut_instance2ended_at()[p.second].second = ended_at;
      TimeLinePushBack(p.second, dev);
      AfterRun(p.second, ended_at);
      sgraph().arc_mgr().InputArc(
          p.second->dst_node(), [&](const TaskArc* arc) {
            const TaskArcInstance* instance_input =
                session().task_arc_instance_mgr().Find(batch, arc);
            mut_tokens().erase(instance_input);
          });
      sgraph().arc_mgr().OutputArc(
          p.second->dst_node(), [&](const TaskArc* arc) {
            const TaskArcInstance* instance_output =
                session().task_arc_instance_mgr().Find(batch, arc);
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
